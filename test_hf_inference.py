from dotenv import load_dotenv
import os
import requests
load_dotenv()  # Load environment variables from .env files

def test_debug():
    token = os.getenv("HUGGING_TOKEN")
    api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-7B-Instruct"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": "what type of ai are you",
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        print("Status code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as exc:
        print("Error:", exc)

if __name__ == "__main__":
    test_debug()
