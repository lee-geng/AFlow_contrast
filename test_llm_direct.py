import json
import os

import requests

# API endpoint
url = "https://api2.aigcbest.top/v1/chat/completions"

# Your API key
api_key = "sk-d3ANninTeNtepnkYZ706Cw9GPKtxO16YXcCU08AEqNgOtEYM"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7,
}

try:
    # Clear proxy-related environment variables for this process.
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(proxy_var, None)

    session = requests.Session()
    session.trust_env = False

    response = session.post(
        url,
        headers=headers,
        json=data,
        timeout=60,
    )

    if response.status_code == 200:
        result = response.json()
        print("请求成功!")
        print(f"模型: {result.get('model')}")
        print(f"回复: {result['choices'][0]['message']['content']}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"网络请求出错: {e}")
except json.JSONDecodeError as e:
    print(f"JSON解析出错: {e}")
except Exception as e:
    print(f"其他错误: {e}")
