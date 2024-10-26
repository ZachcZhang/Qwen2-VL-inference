from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextStreamer, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers import TextIteratorStreamer
from dataclasses import dataclass
from typing import Optional, Union, Any, Sequence, Dict, List, Literal
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

@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]

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
    def _chat(
        self, 
        messages: Sequence[Dict[str, str]]=None,
        input_kwargs: Optional[Dict[str, Any]] = {}
    ) -> List["Response"]:
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
        generation_kwargs = dict(
            **inputs,
            max_new_tokens = input_kwargs["max_new_tokens"],
        )
        generated_ids = self.model.generate(**generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results = []
        for i in range(len(response)):
            eos_index = (generated_ids_trimmed[i] == self.tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(generated_ids_trimmed[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    prompt_length=None,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )
        return results
    

    async def chat(
        self,
        messages: Sequence[Dict[str, str]] = None,
        **input_kwargs,
    ) -> List["Response"]:

        loop = asyncio.get_running_loop()
        input_args = (
            messages,
            input_kwargs,
        )
        async with self._semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._chat, *input_args)
            
    
    @torch.inference_mode()
    def _stream_chat(self, messages:Sequence[dict[str,str]]=None,
                     **input_kwargs):
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
                
        thread = Thread(target=generate_with_no_grad, args=(self.model, generation_kwargs))
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
    await engine.chat()
    
    # async for content in engine.stream_chat():
    #     print(content, end="")

if __name__ == "__main__":
    asyncio.run(main())

    