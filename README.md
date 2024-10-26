# Qwen2-VL-inference

This repository is a `MLLM inference server` which contains a script for inference [qwen2-vl-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) using HuggingFace. 

## Features
- Support OpenAI client
- stream response & normal response


## Update

- [2024/10/26] Open the source code.

## Installation

Install the reqired packages using `environment.yaml`

## Using the `environment.yaml`

```shell
conda env create -f environment.yaml
conda activate qwen2_vl
```

## How to use

- run the server
```shell
API_PORT=10102 python app.py 
```

## Quick start

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://127.0.0.1:10102/v1',
    # required but ignored
    api_key='your_key',
)

stream = client.chat.completions.create(
    messages=[
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
    temperature=0, 
    max_tokens=256,
    stream=True
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        full_response += content
        print(content)
```

## message formart

```python
## url
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

## base64
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

```