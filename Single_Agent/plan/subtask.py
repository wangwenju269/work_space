import sys
sys.path.append('agent')
from typing import Any
from prompt.decompose import  PLANNER_PROMPT_CN, SOLVER_PROMPT_CN , WORKER_PROMPT_CN
from utils.build import GET_TOOL_DESC, Select_tool 
import re
import json5
from pprint import pprint
class  SUBTASK:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\n你是一个任务分解器, 你需要将用户的问题拆分成(最多3个)简单的子任务。{self.im_end}'

    def construct_prompt(self,query,tools_text):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.prefix
        query = PLANNER_PROMPT_CN.format(tool_description = tools_text, query = query)
        query = query.lstrip('\n').rstrip()   
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt
    
    @staticmethod
    def parse_output(message:str):
        '''
        功能：对模型的输出进行解析
        return : 返回子任务的列表和工具函数列表
        '''
        def find_index(message,x_list):
            index_set = []
            position = 0
            for x in x_list:
                position = message.find(x,position )
                index_set.append(position) 
                position += len(x)
            return index_set

        def subtask_map_multi_tools():
            thoughts_index = find_index(message,thoughts)
            action_index =   find_index(message,action_units)
            import bisect
            block = [bisect.bisect_right(action_index,sp) for sp in thoughts_index]
            tool_list = []
            for i in range(len(block)):
                if i == len(block) - 1:
                   tool_list.append(action_units[block[i]:])    
                else:
                   tool_list.append(action_units[block[i]:block[i+1]])    
            assert len(thoughts) == len(tool_list), \
                   'Each Plan should correspond to (zero or one or more) action'
            return thoughts, tool_list  
        
        thoughts = re.findall('Plan: (.+)', message)
        action_units = re.findall('#E\[\d\]\s=\s(.+)', message)
        thoughts,action_units = sorted(set(thoughts), key=thoughts.index), sorted(set(action_units), key = action_units.index)
        return subtask_map_multi_tools()
    
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


    @staticmethod
    def post_process(tool_answer):
        '''
        功能：对工具的解析的结果，进行后处理操作，后续补充。
        '''
        if  not tool_answer:
            tool_answer  = ''  
        stop_list = ["Wolfram Alpha wasn't able to answer it"]
        if   tool_answer in stop_list:
             tool_answer  =  ''  
        return tool_answer 

    
    def sub_task_process(self, query,thoughts, actions, tools_name_text):
        '''
        01 功能：子任务的处理模块,输入thoughts, actions均为LIST,且元素具有对应关系。
        '''
        from collections import defaultdict
        thou_act = defaultdict(str) 
        for thought, action in zip(thoughts, actions):
            if  not action:  # TPTU 拆解子任务没有工具使用 
                thou_act[thought] = "no_use_tool"  
                continue
            '''调用外部的工具返回工具执行的结果  403 是函数名正确，参数解析不正确  404 为函数名错误'''    
            for act in action:
                if 'no_use_tool' in act:    
                    thou_act[thought] = "no_use_tool"  
                    continue  
                Flag = False  # 标志位重置
                try:
                    act_name,input_paremater = self.parse_paremater(act)
                    Flag = True  if act_name in tools_name_text  else False                
                    tool_answer = self.tool_func.call_plugin(act_name,input_paremater)
                    tool_answer = self.post_process(tool_answer)
                    if tool_answer:
                       thou_act[thought] += f"{tool_answer}" if tool_answer.endswith("\n" ) else f"{tool_answer}\n" 
                    else:
                       thou_act[thought] +=  f"_error_403_{act_name}" 
                except  Exception  :
                    if  not thou_act[thought]:  #加入当前 thought 已经有 action 动作时，后续可以省略 
                        if  Flag:  thou_act[thought] +=  f"_error_403_{act_name}"   
                        else:      thou_act[thought] +=  f"_error_404_"  
                        

        print(f"\033[35mObservation:\n{[x for x in  thou_act.values()]}\033[0m")    
        '''
        02 功能：工具返回的结果做归纳总结,错误检查
           说明：
               403 : 将 history 里的信息总结提炼最后一轮QA问答对, 执行 react
               404 : 更改 react 指定思维链的形式 thought
        '''
        history = []
        for thought, action_return in thou_act.items():
            if ('image_url' in action_return):  # 图片的 url 直接返回 
                history.append((thought, action_return.replace('image_url','Final Answer')))
                continue
            if  action_return.startswith('no_use_tool'):
                action_return = query
                for i, (Q, A) in enumerate(history):
                    Q = Q.lstrip('\n').rstrip()
                    A = A.lstrip('\n').rstrip()  
                    action_return += f'\n{Q},{A}'
                action_return += '\n\nRefer to contextual information to answer questions,plase think it step by step'
                answer = self.Summarize(thought, action_return, [])

            elif  action_return.startswith('_error_403_'):
                for i, (Q, A) in enumerate(history):
                    A = self.Summarize(Q, A, history[:i])
                    Q = Q.lstrip('\n').rstrip()
                    A = A.lstrip('\n').rstrip()
                    query = f'{Q}:{A}'
                act_name = action_return.replace('_error_403_','')
                query += f'你必须使用{act_name}工具回答：{thought}'    
                answer = self.sub_task_with_react(query,act_name)

            elif action_return.startswith('_error_404_'):
                sub_query = query
                for Q, A in history:
                    Q = Q.lstrip('\n').rstrip()
                    A = A.lstrip('\n').rstrip()
                    sub_query += f'<|im_end|>\n<|im_start|>assistant\nThought:{Q}\nObservation:{A}'  
                sub_query += f'<|im_end|>\n<|im_start|>assistant\nThought:{thought}\n'    
                answer = self.sub_task_with_react(sub_query, "")
            else:
                  answer =  action_return
            history.append((thought,answer))
        return history
    

    def Summarize(self,thought,action_return, history):
        solver_prompt = SOLVER_PROMPT_CN.format(question = thought, worker_log = action_return)
        answer = self.qwen.qwen_chat(solver_prompt,history)
        return answer
    

    def sub_task_with_react(self,sub_task, act_name):
        from plan.subplan_react import REACT
        react = REACT()
        response =  react(task = sub_task,
                    llm = self.qwen,
                    tool_func = self.tool_func,
                    sub_tool = Select_tool.select_name_tool(act_name) if act_name else self.TOOLS
                    )
        return  response
    
    def conclusion(self,query, history):
        worker_log = ''
        for thought, action_return in history:
            worker_response = WORKER_PROMPT_CN.format(
                              thought=thought,
                              action_resp = action_return)
            worker_log += worker_response
        return self.Summarize(query, worker_log, [])


    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        kwargs:
            task : 原问题
            tool_func  : 调用工具函数集合
            qwen  : llm 语言模型
            tools : 工具描述集合, 
            select_tool: 模型生成的工具集合
            sub_tool : 用户生成的结果
        """

        query =   kwargs.pop('task',None)
        self.tool_func =  kwargs.pop('tool_func',None)
        self.qwen =       kwargs.pop('llm',None)
        select_tool =  kwargs.pop('select_tool',[])
        sub_tool =     kwargs.pop('sub_tool',[])
        if  not select_tool and not sub_tool: 
            from tools.tool import TOOLS 
            self.TOOLS = TOOLS
        else:
            self.TOOLS = select_tool + sub_tool
        print(f"\033[31mOrigin Query:{query}\033[31m")
        tools_text,tools_name_text = GET_TOOL_DESC.get_tools_other_text(self.TOOLS)     
        prompt = self.construct_prompt(query,tools_text)
        turn_id = 0
        while turn_id < 2:
            response = self.qwen.text_completion(prompt, stop_words = ['```'] )     
            response = response.split('```')[0]
            try:  
                   thoughts, actions = [] , [] 
                   thoughts, actions = self.parse_output(response)
                   break
            except Exception :
                   turn_id += 1

        print(f"\033[33mSubtask:{thoughts}\033[0m\n\033[34mTool_func:{list(filter(None,actions))}\033[0m" )
        history = self.sub_task_process(query, thoughts, actions, tools_name_text)
        # print(history)
        finally_answer = self.conclusion(query, history)
        print(f"\033[32mfinally_answer:{finally_answer}\033[0m") 
        return finally_answer 




if __name__ == '__main__':           
    from config.parser import args
    from LLM.Qwen import Qwen  
    from tools.call_plugin import User_defined_tools
    tool_func = User_defined_tools(args.output_file) 
    qwen = Qwen(args.checkpoint)


 

  


    





