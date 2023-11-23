# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# import sys
# sys.path.append('MateAgent')
from  typing import Any
from  typing import List
from  utils.build import GET_TOOL_DESC
from  .prompt import REACT_PROMPT, REFLECTION_HEADER
class React(GET_TOOL_DESC):
    def __init__(self,
                llm,
                External_API,
                ):
        super().__init__()
        self.llm = llm
        self.External_API = External_API

        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.systems = f'{self.im_start}system\nYou are a helpful assistant.{self.im_end}'

        self.reset()

    def reset(self):    
        self.reason_path = ''    # 记录推理路线


    
    def format_reflections(self,reflections: List[str]) -> str:
        if reflections == []:
            return  ''
        else:
            header = REFLECTION_HEADER
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])


    def construct_react_prompt(self, query, history, **kwargs):
        chat_history = history + [(query, '')]  
        im_start, im_end, prompt = self.im_start,  self.im_end, self.systems 
        for i, (query, response) in enumerate(chat_history):
            if (len(chat_history) == 1)  or (i == len(chat_history) - 2) :
                query = REACT_PROMPT.format(
                                    tool_descs = kwargs.pop('tool_desc',''),
                                    tool_names = kwargs.pop('tool_name',''),
                                    reflection = self.format_reflections(kwargs.pop('reflection',[])),
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

    @staticmethod
    def post_process(tool_answer):
        '''
        功能：对工具的解析的结果，进行后处理操作，后续补充。
        '''
        stop_list = ["Wolfram Alpha wasn't able to answer it",'SyntaxError','ModuleNotFoundError','NameError']
        for stop_word in stop_list:
            if stop_word in tool_answer:
               raise Exception("Error calling tool API")
        if  not tool_answer:
            tool_answer  = ''  
        return str(tool_answer) if tool_answer else '' 


    def infer(self, planning_prompt, tool_name):
        # text = ''
        count = 0 
        while count <= 4:
            response = self.llm.text_completion(planning_prompt, stop_words=['Observation:', 'Observation:\n'])
            action, action_input, output = self.parse_latest_plugin_call(response)
            if  (action in tool_name) and action :  # 需要调用插件
                api_output = self.External_API.call_plugin(action.lower(), action_input)
                api_output = str(api_output) if api_output else ''   # 部分api工具返回结果非字符串格式需进行转化后输出
                if "no tool founds" == api_output:
                    break
                print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
                self.reason_path += f'{response} {api_output}\n'
                planning_prompt  +=  f'{response} {api_output}<|im_end|>\n<|im_start|>assistant\n'  # 合并api输出  
            else:  # 生成结束，并且不再需要调用插件
                # text += output 
                break
            count += 1
        # print("\033[32m" + output + "\033[0m")  
        print(f"\033[32m{output}\033[0m")
        output = self.parse_output(output)
        self.reason_path += f'\n{output}'
        return output
    
    
    def run(self, task, **kwargs):
        history = kwargs.pop('history',[])
        select_tool = kwargs.pop('llm_select_tool', False)
        assign_tool = kwargs.pop('user_assign_tool',[])
        tools = self.tool_set((select_tool,task),assign_tool)
        print(f"Origin Query:{task}")

        tool_desc, tool_name = self.get_tool_desc(tools) 
        # print("\033[31m" + query + "\033[0m\n" +
        #         "\033[35m" + 'Candidate tool set  >> ' + tools_name + "\033[0m" )
        planning_prompt  =  self.construct_react_prompt( 
                                                query = task, 
                                                history = history,
                                                tool_desc = tool_desc,
                                                tool_name = tool_name
                                                )
        '''------  获取模型回应 -------'''
        response = self.infer(planning_prompt, tool_name) 
        response = response.lstrip('\n').rstrip() 
        return response
    
if __name__ == '__main__':
   from config.parser import DataArguments
   from llm.model import Qwen
   from tools.plugins import User_defined_tools
   import time
   start = time.time()
   External_API = User_defined_tools() 
   args = DataArguments()
   qwen = Qwen(args.checkpoint)
   react = React(qwen,External_API)  
   task = "画个黑猫吧"       
   react.run(task)
   end = time.time()
   print(end - start)
