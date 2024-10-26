from openai import OpenAI
import sys

client = OpenAI(
    base_url='http://127.0.0.1:10102/v1',
    # required but ignored
    api_key='llamafactory',
)


# 图片base64方式
with open("./00164","rb") as f:
    base64_data = f.read()
    base64_data = str(base64_data,"utf-8")
    
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{base64_data}"
                },
                {"type": "text", "text": "请描述这张图片"},
            ],
        }
    ],
    model='qwen',
    max_tokens=512,
    stream=True
)

print("\n\n===== image base64 stream mode test: =====")
full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        full_response += content
        sys.stdout.write(content)
        sys.stdout.flush()

print("\n\n =====image base64 simple response mode test:===== ")
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{base64_data}"
                },
                {"type": "text", "text": "请描述这张图片"},
            ],
        }
    ],
    model='qwen',
    max_tokens=512,
    stream=False
)
print("\n=====response:",response.choices[0].message.content)


print("\n\n =====image url simple response mode test:===== ")
response = client.chat.completions.create(
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "请详细描述这张图片."},
            ],
        }
    ],
    model='qwen',
    max_tokens=512,
    stream=False
)
print("\n=====response:",response.choices[0].message.content)
