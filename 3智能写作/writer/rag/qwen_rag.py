#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# 使用 qwen_agent里 的 rag 检索模式, 客户端脚本代码如下。
from fastapi import FastAPI, Request
from qwen_agent.agents.doc_qa import ParallelDocQA
from typing import List, Dict, Any, Union
app = FastAPI()
async def rag(messages:List[Dict[str, Any]]):
    bot = ParallelDocQA(
                llm = {
                    "model": "qwen",
                    "model_server": "http://10.9.27.51:7800/v1",
                    "generate_cfg": {
                        "max_retries": 10
                                     }
                    } 
                    ) 
    for rsp in bot.run(messages):
        continue
    return rsp[0]['content']
    
@app.post("/v1")
async def handle_rag(request: Request):
    data = await request.json()
    messages = data.get('messages')
    response = await rag(messages)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

"""

import os
import requests
from datetime import datetime
url = "http://10.9.27.51:6000/v1"
headers = {
            "Content-Type": "application/json"
          }
def request_body(func):  
    def wrapper(**kwargs):
        text = kwargs.pop('text','')
        file_folder =   kwargs.pop('file_folder','')
        if not file_folder.exists():
           return  ''
        file_name_list = os.listdir(file_folder) 
        if not text or not file_name_list:
            return ''
        content = [{'text': text}]
        for file_name in file_name_list:
            file = str(file_folder / f'{file_name}')
            content.append({'file':file})
        data = {"messages": [{ "role": "user",  "content":content}]}
        start_time = datetime.now()
        contexts = func(data)  
        end_time = datetime.now()
        print(f"执行时间：{(end_time - start_time).total_seconds()}秒")
        print(contexts)
        return contexts             
    return wrapper

@request_body
def qwen_request(data):
    response = requests.post(url, headers=headers, json=data).json()
    return response['response']
