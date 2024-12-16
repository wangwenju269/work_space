import openai

class OpenAi:
    def __init__(self, cfg):
        self.model = cfg.get('model', 'gpt-3.5-turbo')
        self.api_base = cfg.get('api_base', 'https://api.openai.com/v1')
        self.agent_type = 'react'

    def generate(self,
                 llm_artifacts,
                 functions=None,
                 function_call=None,
                 **kwargs):
        """
        生成 OpenAI 响应。

        Args:
            llm_artifacts: 用户输入的内容。
            functions: 可选的函数列表。
            function_call: 可选的函数调用策略。
            **kwargs: 其他参数。

        Returns:
            str: 生成的响应内容。
        """
        if functions is None:
            functions = []

        messages = [{'role': 'user', 'content': llm_artifacts}]

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                api_base=self.api_base,
                messages=messages,
                stream=False
            )
        except Exception as e:
            print(f'Input: {messages}, Original error: {str(e)}')
            raise e

        # 截取响应内容
        content = response["choices"][0]["message"]["content"]
        return content