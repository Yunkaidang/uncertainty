import re
import os
import sys
import json
import time
import base64
import requests
import traceback
from zhipuai import ZhipuAI
import openai
from openai import OpenAI
from dashscope import MultiModalConversation
class Chat_gpt4v:
    def __init__(self, model="", timeout_sec=20):
        self.model = model
        self.timeout = timeout_sec
        if self.model in ['qwen-vl-plus','qwen-vl-max']:
            self.api_key = 'sk-be9b4bbde474f7dfgv......'

        elif self.model in ['glm-4v']:
            self.client = ZhipuAI(api_key="163f3.....")


        else :
            self.api_key = 'sk-E.....'
            self.client = OpenAI(
                api_key=self.api_key,
                base_url='https://...',
            )


    def chat_completion(self, messages, temperature=0.1, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0, plain_use=False):
        if plain_use:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )

        return response

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_eval_plain_use_gpt4v(chat, content, image_path,temperature,
             max_tokens=1024,
             fail_limit=1,
             return_resp=False):

    if chat.model in ['qwen-vl-plus','qwen-vl-max']:
        print("file://"+image_path)
        fail_cnt = 0
        while True:
            try:
                resp= MultiModalConversation.call(model=chat.model ,
                    api_key=os.getenv('sk-87...'),                 
                    
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"image" : "file://"+image_path,},
                                {"text": content},
                               ],
                        }
                    ],
                )
                print(resp)
                try:
                    content = resp["output"]["choices"][0]["message"]["content"][0]['text']
                    time.sleep(2)
                    return [content]
                except:
                    print(f'Response: {resp}')
                    
            except Exception as e:
                print(e)
            fail_cnt += 1
            if fail_cnt == fail_limit:
                return f'-1\n<no_response>'
            time.sleep(10 + fail_cnt)
    elif chat.model in ['glm-4v']:
        base64_image = encode_image(image_path)
        fail_cnt = 0
        while True:
            try:
                response = chat.client.chat.completions.create(
                    model="glm-4v",
                    messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": content},
                                        {
                                            "type": "image_url",
                                            "image_url" : {
                                                "url": f"data:image/jpeg;base64,{base64_image}",
                                                "detail":"high"
                                            },
                                        },
                                    ],
                                }
                            ],
                
                )
                print(response.choices[0].message.content)
                try:
                    content = response.choices[0].message.content
                    time.sleep(2)
                    return [content]
                except:
                    print(f'Response: {response}')      
            except Exception as e:
                print(e)
            fail_cnt += 1
            if fail_cnt == fail_limit:
                return f'-1\n<no_response>'
            time.sleep(10 + fail_cnt)
    elif image_path=="0":
        fail_cnt = 0
        while True:
            try:
                resp = chat.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": content}
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    plain_use=True
                )
                try:
                    content = resp.choices[0].message.content
                    time.sleep(2)
                    if return_resp:
                        return content, resp
                    else:
                        return [content]
                except:
                    print(f'Response_except: {resp}')
            except Exception as e:
                print(e)
            fail_cnt += 1
            if fail_cnt == fail_limit:
                return f'-1\n<no_response>'
            time.sleep(10 + fail_cnt)

    else:
        base64_image = encode_image(image_path)
        fail_cnt = 0
        while True:
            try:
                resp = chat.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": content},
                                {
                                    "type": "image_url",
                                    "image_url" : {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail":"high"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    plain_use=True
                )
                try:
                    content = resp.choices[0].message.content
                    # print(f'Response_try: {resp}')
                    # print(f'content',content)
                    # print('success!', flush=True)
                    time.sleep(2)
                    if return_resp:
                        return content, resp
                    else:
                        return [content]
                except:
                    print(f'Response_except: {resp}')
            except Exception as e:
                print(e)
            fail_cnt += 1
            if fail_cnt == fail_limit:
                return f'-1\n<no_response>'
            time.sleep(10 + fail_cnt)



