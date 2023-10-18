import sys
sys.path.append('agent')
from typing import Any
from utils.build import GET_TOOL_DESC, Select_tool 
from .check_parameters import CHECK_PROMPT, EXAMPLE
import re, json5  
class CHECK:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.systems = f'{self.im_start}system\n你是个工具函数参数解析器{self.im_end}'

    def construct_prompt(self, query, kwargs):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.systems 
        tools_name = kwargs['tools_name']  
        logging = ''
        for i,(sub_query, sub_response) in enumerate(kwargs['history']):
            logging += f'\n{i+1}. {sub_query}-->{sub_response}'
        query = CHECK_PROMPT.format(
                                    EXAMPLES = EXAMPLE[tools_name],
                                    tool_description = kwargs['tools_desc'],
                                    cur_question = query,
                                    history =  logging
                                    )
        query = query.lstrip('\n').rstrip()              # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt 
    
    @staticmethod
    def parse_paremater(inputs:str):
        '''
        功能：将输入解析为函数名，函数参数形式
        return : 函数名，函数参数
        ''' 
            # input_paremater = re.findall('"([^"]*)"', act.strip())[0].replace('\\n', '\n')
            # input_paremater = TOOL._construct_Input(act_name,input_paremater)
        act_name, input_paremater =  re.findall(r'(.*?)\((.*)\)',inputs.strip())[0]
        index = input_paremater.index('=')
        key = input_paremater[:index].strip()
        value = input_paremater[(index+1):].strip()
        input_paremater = json5.dumps({key:eval(value)},ensure_ascii=False)
        return act_name, input_paremater

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        query = kwargs.pop('task',None)
        tool_func = kwargs.pop('tool_func',None)
        qwen = kwargs.pop('llm',None)
        select_tool = kwargs.pop('select_tool',[])
        sub_tool = kwargs.pop('sub_tool',[])
        if not select_tool and not sub_tool: 
            from tools.tool import TOOLS 
        else:
            TOOLS = select_tool + sub_tool  
        tools_desc, tools_name = GET_TOOL_DESC.get_tools_text(TOOLS) 
        kwargs.update(tools_desc = tools_desc,tools_name = tools_name)
        planning_prompt  =  self.construct_prompt(query,kwargs)
        response = qwen.text_completion(planning_prompt) 
        act_name, input_paremater = self.parse_paremater(response)  
        tool_answer = tool_func.call_plugin(act_name,input_paremater)
        print(f"\033[31mCheck:\n{response}-->response:{tool_answer}\033[31m")
        return tool_answer
    
if __name__ == '__main__':
    from config.parser import args
    from LLM.Qwen import Qwen  
    from tools.call_plugin import User_defined_tools
    tool_func = User_defined_tools() 
    qwen = Qwen(args.checkpoint)
    check = CHECK()
    check(
        task = '百度总裁年龄的三次幂减去阿里执行官年龄的平方,该数值的两倍是多少?',
        history = [('搜索百度总裁比阿里执行官的年龄信息','百度总裁52岁,阿里执行官35岁')],
        tool_func = tool_func,
        select_tool = Select_tool.select_name_tool('math') ,
        llm = qwen,
    )   