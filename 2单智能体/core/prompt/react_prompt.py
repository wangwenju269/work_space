import json

from .prompt import LengthConstraint, PromptGenerator

# 默认的 REACT 系统模板
REACT_DEFAULT_SYSTEM_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

<user_tool_list>"""

# 默认的 REACT 指令模板
REACT_DEFAULT_INSTRUCTION_TEMPLATE = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [<tool_names>]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)

Final Answer: the final answer can successfully solve the original question.

Begin!
"""

# 默认的用户输入模板
REACT_DEFAULT_USER_TEMPLATE = """Question: <user_input>\n"""

# 默认的执行结果模板
REACT_DEFAULT_EXEC_TEMPLATE = """ <exec_result>\n"""

# 工具描述模板
TOOL_DESC = (
    "{name_for_model}: Call this tool to interact with the {name_for_human} API. "
    "What is the {name_for_human} API useful for? {description_for_model} "
    "Parameters: {parameters}"
)

# 格式描述
FORMAT_DESC = {
    'json': 'Format the arguments as a JSON object.',
    'code': 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
}


class ReactPromptGenerator(PromptGenerator):
    """
    ReactPromptGenerator 类，用于生成基于 REACT 框架的提示。
    """

    def __init__(self,
                 system_template=REACT_DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template=REACT_DEFAULT_INSTRUCTION_TEMPLATE,
                 user_template=REACT_DEFAULT_USER_TEMPLATE,
                 exec_template=REACT_DEFAULT_EXEC_TEMPLATE,
                 assistant_template='',
                 sep='\n\n',
                 llm=None,
                 length_constraint=LengthConstraint(),
                 **kwargs):
        """
        初始化 ReactPromptGenerator。

        Args:
            system_template (str): 系统模板。
            instruction_template (str): 指令模板。
            user_template (str): 用户输入模板。
            exec_template (str): 执行结果模板。
            assistant_template (str): 助手响应模板。
            sep (str): 内容分隔符。
            llm: LLM 实例。
            length_constraint (LengthConstraint): 长度约束。
            **kwargs: 其他参数。
        """
        super().__init__(
            system_template=system_template,
            instruction_template=instruction_template,
            user_template=user_template,
            exec_template=exec_template,
            assistant_template=assistant_template,
            sep=sep,
            llm=llm,
            length_constraint=length_constraint,
            **kwargs
        )

    def init_prompt(self, task, tool_list, knowledge_list, **kwargs):
        """
        初始化提示。

        Args:
            task (str): 用户任务。
            tool_list (list): 工具列表。
            knowledge_list (list): 知识库列表。
            **kwargs: 其他参数。
        """
        if len(self.history) == 0:
            super().init_prompt(task, tool_list, knowledge_list, **kwargs)

            # 生成工具描述字符串
            self.tool_str = self.get_tool_str(tool_list)
            tool_names = [f"'{str(tool.name)}'" for tool in tool_list]
            self.tool_names = ','.join(tool_names)

            # 系统角色状态
            system_role_status = kwargs.get('system_role_status', False)
            if system_role_status:
                system_message = {
                    'role': 'system',
                    'content': self.system_prompt
                }
                self.history.insert(0, system_message)
            else:
                self.history[0]['content'] = self.system_prompt + self.history[0]['content']
        else:
            # 添加用户输入和助手响应到历史记录
            self.history.append({
                'role': 'user',
                'content': self.user_template.replace('<user_input>', task)
            })
            self.history.append({
                'role': 'assistant',
                'content': self.assistant_template
            })

    def get_tool_str(self, tool_list):
        """
        生成工具描述字符串。

        Args:
            tool_list (list): 工具列表。

        Returns:
            str: 工具描述字符串。
        """
        tool_texts = []
        for tool in tool_list:
            tool_texts.append(
                TOOL_DESC.format(
                    name_for_model=tool.name,
                    name_for_human=tool.name,
                    description_for_model=tool.description,
                    parameters=json.dumps(tool.parameters, ensure_ascii=False)
                ) + ' ' + FORMAT_DESC['json']
            )
        tool_str = '\n\n'.join(tool_texts)
        return tool_str

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
            self.history[-1]['content'] += llm_result
        if exec_result:
            exec_result = self.exec_template.replace('<exec_result>', str(exec_result))
            self.history[-1]['content'] += exec_result

        self.prompt = self.preprocessor(self.history)
        return self.prompt

    def preprocessor(self, sources):
        """
        预处理历史记录，生成最终的提示。

        Args:
            sources (list): 历史记录列表。

        Returns:
            str: 生成的提示。
        """
        prompt = ''
        response = ''
        for i in range(0, len(sources), 2):
            if sources[i]['role'] == 'user':
                query = sources[i]['content']
                # 每轮工具交互完成后，工具集合会发生变化。
                query = query.replace('<tool_names>', self.tool_names).replace('<user_tool_list>', self.tool_str)
            if sources[i + 1]['role'] == 'assistant':
                response = sources[i + 1]['content']
            prompt += (query + '<|im_end|>\n<|im_start|>assistant\n' + response) if response else query
        return prompt