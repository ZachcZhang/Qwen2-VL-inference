## restful request
``` python
import requests
import sys
import json

def fetch_qwen2vl(user_info):
    endpoint = "http://127.0.0.1:10102/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
    }
    payload = json.dumps({
        "model": "qwen2vl",
        "messages": [
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
        "stream":True,
        "max_tokens":512
        
    })
    response =  requests.post(endpoint, headers=headers, data=payload, stream=True)
    for chunk in response.iter_lines(): 
        if chunk:
            decoded_line = chunk.decode('utf-8')
            data = decoded_line[len("data: "):].strip()
            try:
                data = json.loads(data)
                choieces = data["choices"][0]
                if choieces["finish_reason"] is not None:
                    break
                sys.stdout.write(choieces["delta"]["content"])
                sys.stdout.flush()
                
            except Exception as e:
                pass

def main():
    question = """
    请描述图片
    """
    fetch_qwen2vl(question)
    
if __name__ == "__main__":
    main()

```