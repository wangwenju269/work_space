import copy
from typing import Union

LANG = 'zh'

# 根据语言设置提示模板
if LANG == 'zh':
    KNOWLEDGE_PROMPT = '# 知识库'
    KNOWLEDGE_INTRODUCTION_PROMPT = '以下是我上传的文件“<file_name>”的内容:'
else:
    KNOWLEDGE_PROMPT = '# Knowledge Base'
    KNOWLEDGE_INTRODUCTION_PROMPT = 'The following is the content of the file "<file_name>" I uploaded:'

KNOWLEDGE_CONTENT_PROMPT = """```
<knowledge_content>
```"""

DEFAULT_PROMPT_INPUT_LENGTH_MAX = 999999999999


class LengthConstraint:
    """
    用于限制输入和知识库内容长度的类。
    """

    def __init__(self):
        self.knowledge = DEFAULT_PROMPT_INPUT_LENGTH_MAX
        self.input = DEFAULT_PROMPT_INPUT_LENGTH_MAX
        self.prompt_max_length = 10000

    def update(self, config: dict):
        """
        更新长度限制配置。

        Args:
            config (dict): 包含长度限制配置的字典。
        """
        if config is not None:
            self.knowledge = config.get('knowledge', self.knowledge)
            self.input = config.get('input', self.input)
            self.prompt_max_length = config.get('prompt_max_length', self.prompt_max_length)


class PromptGenerator:
    """
    用于生成提示的类。
    """

    def __init__(self,
                 system_template: str = '',
                 instruction_template: str = '',
                 user_template: str = '<user_input>',
                 exec_template: str = '',
                 assistant_template: str = '',
                 sep='\n\n',
                 llm=None,
                 length_constraint=LengthConstraint(),
                 **kwargs):
        """
        初始化 PromptGenerator。

        Args:
            system_template (str): 系统模板，通常是 LLM 的角色。
            instruction_template (str): 指示 LLM 的指令。
            user_template (str): 用户输入的前缀。
            exec_template (str): 执行结果的包装字符串。
            assistant_template (str): 助手响应的前缀。
            sep (str): 内容分隔符。
            length_constraint (LengthConstraint): 内容长度限制。
        """
        self.system_template = system_template
        self.instruction_template = instruction_template
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.exec_template = exec_template
        self.sep = sep
        self.prompt_max_length = length_constraint.prompt_max_length
        self.reset()

    def reset(self):
        """
        重置提示和历史记录。
        """
        self.prompt = ''
        self.history = []
        self.messages = []

    def init_prompt(self,
                    task,
                    tool_list,
                    knowledge_list,
                    llm_model=None,
                    **kwargs):
        """
        初始化提示。

        Args:
            task (str): 用户任务。
            tool_list (list): 工具列表。
            knowledge_list (list): 知识库列表。
            llm_model (str): LLM 模型名称。
            **kwargs: 其他参数。
        """
        prompt = self.sep.join([self.system_template, self.instruction_template])
        prompt += '<knowledge><history>'

        # 生成知识库字符串
        knowledge_str = self.get_knowledge_str(
            knowledge_list, file_name=kwargs.get('file_name', ''))
        prompt = prompt.replace('<knowledge>', knowledge_str)

        # 生成工具描述字符串
        tool_str = self.get_tool_str(tool_list)
        prompt = prompt.replace('<tool_list>', tool_str)

        # 生成历史记录字符串
        history_str = self.get_history_str()
        prompt = prompt.replace('<history>', history_str)

        self.system_prompt = copy.deepcopy(prompt)

        # 用户输入
        user_input = self.user_template.replace('<user_input>', task)
        prompt += f'{self.sep}{user_input}'

        # 助手输入
        prompt += f'{self.sep}{self.assistant_template}'

        # 存储历史记录
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'assistant', 'content': self.assistant_template})

        self.prompt = prompt

        # 生成函数调用列表
        self.function_calls = self.get_function_list(tool_list)

    def generate(self, llm_result, exec_result: Union[str, dict]):
        """
        生成下一轮提示。

        Args:
            llm_result (str): LLM 的响应结果。
            exec_result (Union[str, dict]): 执行结果。

        Returns:
            str: 生成的提示。
        """
        if isinstance(exec_result, dict):
            exec_result = str(exec_result['result'])
        return self._generate(llm_result, exec_result)

    def _generate(self, llm_result, exec_result: str):
        """
        基于之前的 LLM 结果和执行结果生成下一轮提示，并更新历史记录。

        Args:
            llm_result (str): LLM 的响应结果。
            exec_result (str): 执行结果。

        Returns:
            str: 生成的提示。
        """
        if llm_result:
            self.prompt += llm_result
            self.history[-1]['content'] += llm_result
        if exec_result:
            exec_result = self.exec_template.replace('<exec_result>', str(exec_result))
            self.prompt += f'{self.sep}{exec_result}'
            self.history[-1]['content'] += f'{self.sep}{exec_result}'

        return self.prompt

    def _generate_messages(self, llm_result, exec_result: str):
        """
        生成下一轮提示并更新历史记录。

        Args:
            llm_result (dict): LLM 的响应结果。
            exec_result (str): 执行结果。

        Returns:
            list: 更新后的历史记录。
        """
        if not llm_result and not exec_result:
            return self.history

        function_call = llm_result.get('function_call', None)
        if function_call is not None:
            llm_result['content'] = ''
        self.history.append(llm_result)

        if exec_result and function_call:
            exec_message = {
                'role': 'function',
                'name': 'execute',
                'content': exec_result,
            }
            self.history.append(exec_message)

        return self.history

    def get_tool_str(self, tool_list):
        """
        生成工具列表字符串。

        Args:
            tool_list (list): 工具列表。

        Returns:
            str: 工具列表字符串。
        """
        tool_str = self.sep.join([f'{i + 1}. {t}' for i, t in enumerate(tool_list)])
        return tool_str

    def get_function_list(self, tool_list):
        """
        从工具列表生成函数调用列表。

        Args:
            tool_list (list): 工具列表。

        Returns:
            list: 函数调用列表。
        """
        functions = [tool.get_function() for tool in tool_list]
        return functions

    def get_knowledge_str(self,
                          knowledge_list,
                          file_name='',
                          only_content=False,
                          **kwargs):
        """
        生成知识库字符串。

        Args:
            knowledge_list (list): 知识库列表。
            file_name (str): 文件名。
            only_content (bool): 是否只返回内容部分。

        Returns:
            str: 知识库字符串。
        """
        knowledge = self.sep.join([f'{i + 1}. {k}' for i, k in enumerate(knowledge_list)])
        knowledge_content = KNOWLEDGE_CONTENT_PROMPT.replace('<knowledge_content>', knowledge)

        if only_content:
            return knowledge_content
        else:
            knowledge_introduction = KNOWLEDGE_INTRODUCTION_PROMPT.replace('<file_name>', file_name)
            knowledge_str = f'{KNOWLEDGE_PROMPT}{self.sep}{knowledge_introduction}{self.sep}{knowledge_content}' if knowledge_list else ''
            return knowledge_str

    def get_history_str(self):
        """
        生成历史记录字符串。

        Returns:
            str: 历史记录字符串。
        """
        history_str = ''
        for i in range(len(self.history)):
            history_item = self.history[len(self.history) - i - 1]
            text = history_item['content']
            if len(history_str) + len(text) + len(self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str