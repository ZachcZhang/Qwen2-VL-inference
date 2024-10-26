from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers import TextIteratorStreamer

from typing import Optional, Union, Any, Sequence, Dict
from threading import Thread

import asyncio
import concurrent.futures
import os
import torch

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")



class Qwen2VL():
    def __init__(self, model:Optional[str] = "Qwen/Qwen2-VL-7B-Instruct"):
        # Qwen2VLForConditionalGeneration
        self.model = AutoModelForVision2Seq.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 1)))

    def _process_args(self, messages: Sequence[Dict[str, str]]
        ) -> Sequence[Dict[str, str]]:
        processed_msg = []
        for msg in messages:
            msg = msg.model_dump()
            for detail in msg["content"]:
                if detail["image"] is None:
                    detail.pop("image")
            processed_msg.append(msg)
        return processed_msg
    
    @torch.inference_mode()
    def _stream_chat(self, messages:Sequence[dict[str,str]]=None,
                     **input_kwargs):
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        #             },
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ]
        messages = self._process_args(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=input_kwargs["max_new_tokens"],
        )
        def generate_with_no_grad(model, generation_kwargs):
            with torch.no_grad():
                model.generate(**generation_kwargs)
                
        thread = Thread(target=generate_with_no_grad, args=(self.model,generation_kwargs))
        thread.start()
        
        def stream():
            try:
                return streamer.__next__()
            except StopIteration:
                raise StopAsyncIteration()

        return stream

    @torch.inference_mode()
    async def stream_chat(self,
                        messages:Sequence[dict[str,str]]=None,
                        system: Optional[str] = None,
                        tools: Optional[str] = None,
                        image = None,
                        **input_kwargs,
                        ):
        loop = asyncio.get_running_loop()
        async with self._semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stream = self._stream_chat(messages, **input_kwargs)
                while True:
                    try:
                        yield await loop.run_in_executor(pool, stream)
                    except StopAsyncIteration:
                        break
                    
    async def start(self) -> None:
        self._semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 1)))
                    
# 主函数使用 asyncio.run 启动异步代码
async def main():
    engine = Qwen2VL()
    async for content in engine.stream_chat():
        print(content, end="")

if __name__ == "__main__":
    asyncio.run(main())

    