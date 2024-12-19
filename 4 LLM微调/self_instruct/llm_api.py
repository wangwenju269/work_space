import requests
import time
from typing import Any
from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback

API_KEYS = {
    "deepseek": '',
    "zhipu": '',
    "anthropic": '',
    'vllm': 'token-abc123',
}

API_URLS = {
    'deepseek': 'https://api.deepseek.com/v1/chat/completions',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    "anthropic": 'https://api.anthropic.com/v1/messages',
    'vllm': 'http://10.9.27.51:7800/v1/chat/completions',
}


import time
import requests
from typing import Any

class UserLLM(CustomLLM):
    model: str  # 模型名称(deepseek,vllm,...)
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 4096  # 最大输出 token 数量
    
    @property
    def metadata(self) -> LLMMetadata:
        """获取 LLM 元数据"""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: Any = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        完成 LLM 请求
        :param prompt: 输入提示
        :param temperature: 温度参数
        :param max_new_tokens: 最大生成 token 数量
        :param stop: 停止序列
        :param kwargs: 其他参数
        :return: 完成响应
        """
        # 根据模型名称选择 API 密钥和 URL
        if 'deepseek' in self.model:
            api_key, api_url = API_KEYS['deepseek'], API_URLS['deepseek']
        elif 'glm' in self.model:
            api_key, api_url = API_KEYS['zhipu'], API_URLS['zhipu']
        elif 'claude' in self.model:
            api_key, api_url = API_KEYS['anthropic'], API_URLS['anthropic']
        else:
            api_key, api_url = API_KEYS['vllm'], API_URLS['vllm']
        
        # 构造消息列表
        messages = [{'role': 'user', 'content': prompt}]
        tries = 0

        # 重试机制，最多重试 5 次
        while tries < 5:
            tries += 1
            try:
                # 根据模型类型设置请求头
                if 'claude' not in self.model:
                    headers = {
                        'Authorization': "Bearer {}".format(api_key),
                    }
                else:
                    headers = {
                        'x-api-key': api_key,
                        'anthropic-version': "2023-06-01",
                    }
                
                # 发送请求
                resp = requests.post(
                    api_url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "stop" if 'claude' not in self.model else 'stop_sequences': stop,
                    },
                    headers=headers,
                    timeout=600,  # 设置超时时间为 600 秒
                )

                # 检查响应状态码
                if resp.status_code != 200:
                    raise Exception(resp.text)

                # 解析响应
                resp = resp.json()
                return CompletionResponse(text=resp["choices"][0]["message"]["content"])

            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                # 处理特定错误
                if "maximum context length" in str(e):
                    raise e
                elif "triggering" in str(e):
                    return 'Trigger OpenAI\'s content management policy.'

                # 打印错误信息并重试
                print("Error Occurs: \"%s\"        Retry ..." % (str(e)))
                time.sleep(1)

        # 如果重试 5 次后仍失败，返回空响应
        return CompletionResponse(text='')
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """
        流式完成 LLM 请求
        :param prompt: 输入提示
        :param kwargs: 其他参数
        :return: 流式完成响应生成器
        """
        raise NotImplementedError("stream_complete is not implemented.")


if __name__ == '__main__':
    # 示例：使用 deepseek-chat 模型进行请求
    model = 'deepseek-chat'
    prompt = '你是谁'
    query_llm = UserLLM(model=model)
    output = query_llm.complete(prompt)
    print(output)