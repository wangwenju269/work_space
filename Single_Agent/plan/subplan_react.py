from typing import Any
from utils.build import GET_TOOL_DESC
from prompt.react import REACT_PROMPT
  
class REACT:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.systems = f'{self.im_start}system\nYou are a helpful assistant.{self.im_end}'

    def construct_prompt(self, 
                         query,
                         history,
                         **kwargs):
        chat_history = history + [(query, '')]  
        im_start, im_end, prompt = self.im_start,  self.im_end, self.systems 
        for i, (query, response) in enumerate(chat_history):
            if (len(chat_history) == 1)  or (i == len(chat_history) - 2) :
                query = REACT_PROMPT.format(
                                    tool_descs = kwargs['tools_text'],
                                    tool_names = kwargs['tools_name_text'],
                                    query = query
                                    )
            query = query.lstrip('\n').rstrip()              # 重要！若不 strip 会与训练时数据的构造方式产生差异。
            response = response.lstrip('\n').rstrip()         # 重要！若不 strip 会与训练时数据的构造方式产生差异。
            prompt += f"\n{im_start}user\n{query}{im_end}"
            prompt += f"\n{im_start}assistant\n{response}{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt 
    
    def parse_latest_plugin_call(self,text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:') : j].strip()
            plugin_args = text[j + len('\nAction Input:') : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text
    
    def parse_output(self,text):
        start_flag = 'Thought:'
        stop_flag = 'Final Answer:'
        if  stop_flag in text:
            i = text.rfind(stop_flag)
            finally_answer = text[i+len(stop_flag):] 
            finally_answer = finally_answer.lstrip(' ').rstrip('\n')  
            return finally_answer
        else:
            return text
    
    def infer(self, planning_prompt,tools_name,llm,TOOL):
        # text = ''
        count = 0 
        while count <= 2:
            response = llm.text_completion(planning_prompt, stop_words=['Observation:', 'Observation:\n'])
            action, action_input, output = self.parse_latest_plugin_call(response)
            if  (action in tools_name) and action :  # 需要调用插件
                api_output = TOOL.call_plugin(action.lower(), action_input)
                api_output = str(api_output) if api_output else ''   # 部分api工具返回结果非字符串格式需进行转化后输出
                if "no tool founds" == api_output:
                    break
                print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
                planning_prompt = planning_prompt + response + ' ' + api_output # 合并api输出  
                planning_prompt += '<|im_end|>'
                planning_prompt += "\n<|im_start|>assistant\n"
                # text +=  output + f'\nObservation: {api_output}\n'
            else:  # 生成结束，并且不再需要调用插件
                # text += output 
                break
            count += 1
        # print("\033[32m" + output + "\033[0m")  
        answer = self.parse_output(output)
        return answer
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        query = kwargs.pop('task',None)
        tool_func = kwargs.pop('tool_func',None)
        qwen = kwargs.pop('llm',None)
        select_tool = kwargs.pop('select_tool',[])
        sub_tool = kwargs.pop('sub_tool',[])
        history = kwargs.pop('history',[])
        if not select_tool and not sub_tool: 
            from tools.tool import TOOLS 
        else:
            TOOLS = select_tool + sub_tool  
        tools_text, tools_name = GET_TOOL_DESC.get_tools_text(TOOLS) 
        # print("\033[31m" + query + "\033[0m\n" +
        #         "\033[35m" + 'Candidate tool set  >> ' + tools_name + "\033[0m" )
        planning_prompt  =  self.construct_prompt( 
                                                query = query, 
                                                history = history,
                                                tools_text = tools_text,
                                                tools_name_text = tools_name
                                                )
        '''------  获取模型回应 -------'''
        response = self.infer(planning_prompt,tools_name, qwen, tool_func) 
        response = response.lstrip('\n').rstrip() 
        return response