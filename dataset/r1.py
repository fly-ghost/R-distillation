"""
此文件用于数据增强, 使用Deepseek-r1生成数据集
"""
import sys

import os

# 导入所需的库
from openai import OpenAI

# 设置 OpenAI API路径
API_BASE = "https://sg.uiuiapi.com/v1"
# 设置 OpenAI API 密钥
API_KEY = "sk-ogFbxgNBupDzZhQJTPt0RqXGhWkSFs6MT0rRszkRpbrlWD1M"

from openai import OpenAI

client = OpenAI(
  api_key=API_KEY,
  base_url=API_BASE
)

def get_completion(prompt, model="deepseek-r1"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only provides code completions without reasoning."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0.95,

    )
    return response.choices[0].message.content

prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
print(get_completion(prompt))
