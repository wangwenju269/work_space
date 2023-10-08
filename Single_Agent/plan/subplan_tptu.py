import sys
sys.path.append('agent')
from typing import Any
from prompt.tptu import  PLANNER_PROMPT_CN, SOLVER_PROMPT_CN , WORKER_PROMPT_CN
from utils.build import GET_TOOL_DESC  
import re
import json5
from pprint import pprint
class  TPTU:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\n你是一个任务分解器, 你需要将用户的问题拆分成简单的子任务。{self.im_end}'

    def construct_prompt(self,query,tools_text):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.prefix
        query = PLANNER_PROMPT_CN.format(tool_description = tools_text, query = query)
        query = query.lstrip('\n').rstrip()   
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt
    
    def parse_output(self, message):
        def subtask_map_multi_tools():
            thoughts_index = [message.find(x) for x in thoughts]
            action_index =   [message.find(x) for x in action_units]
            import bisect
            block = []
            for sp in thoughts_index:
                i = bisect.bisect_right(action_index,sp)
                block.append(i)
            tool_list = []
            for i in range(len(block)):
                if i == len(block) - 1:
                   tool_list.append(action_units[block[i]:])    
                else:
                   tool_list.append(action_units[block[i]:block[i+1]])       
            return thoughts, tool_list  
        
        thoughts = re.findall('Plan: (.+)', message)
        action_units = re.findall('#E\[\d\]\s=\s(.+)', message)
        return subtask_map_multi_tools()
        
    def format_solver(self, thought_action_responses,qwen) :
        history = []
        for thought, action_return in thought_action_responses.items():
            if ('image_url' in action_return):
               history.append((thought, action_return.replace('image_url','Final Answer')))
               continue
            solver_prompt = SOLVER_PROMPT_CN.format(question = thought, worker_log = action_return)
            answer, _ = qwen.model.chat(tokenizer = qwen.tokenizer,
                                    query = solver_prompt ,
                                    history = history,
                                    append_history = False)
            history.append((thought,answer))
        return history
    
    def post_process(self,tool_answer):
        if  not tool_answer:
            print( "\033[34mA:" + '文件保存在：/workspace/temp' + "\033[0m" )
            tool_answer  =  ''  
        elif "Wolfram Alpha wasn't able to answer it" in tool_answer:
              tool_answer  =  '' 
        else:
            tool_answer =  tool_answer if tool_answer.endswith('\n') else tool_answer +'\n'
        return tool_answer 
    

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        query = kwargs.pop('task',None)
        TOOL =  kwargs.pop('TOOL',None)
        qwen =  kwargs.pop('LLM',None)
        task_map_tool =  kwargs.pop('tool_query_func',None)
        if task_map_tool:
            tools = task_map_tool(query,k = 4)  
        else:
            from tools.tool import tools   
        tools_text,tools_name_text = GET_TOOL_DESC.get_tools_other_text(tools)     
        prompt = self.construct_prompt(query,tools_text)
        thoughts, actions = [],[] 
        turn_id = 0
        while turn_id < 2:
            response = qwen.text_completion(prompt, stop_words = ['```'] )
            response = response.split('```')[0]
            try:
                   thoughts, actions = self.parse_output(response)
                   break
            except Exception :
                   turn_id += 1
   
        print(f"\033[33mthought:{thoughts}\033[0m\n\033[34mactions:{actions}\033[0m" )

        from collections import defaultdict
        thought_action_responses = defaultdict(str)  
        for thought, action in zip(thoughts, actions):
            for act in action:
                try:
                    act_name, input_paremater =  re.findall(r'(.*?)\((.*)\)',act.strip())[0]
                       # input_paremater = re.findall('"([^"]*)"', act.strip())[0].replace('\\n', '\n')
                       # input_paremater = TOOL._construct_Input(act_name,input_paremater)
                    """工具参数解析"""
                    index = input_paremater.index('=')
                    key = input_paremater[:index].strip()
                    value = input_paremater[(index+1):].strip()
                    input_paremater = json5.dumps({key:eval(value)},ensure_ascii=False)

                    func = TOOL.config[act_name]
                    tool_answer = func(input_paremater)    
                    tool_answer = self.post_process(tool_answer)
                    thought_action_responses[thought] +=  f"{tool_answer}"             \
                                                          if tool_answer.endswith("\n" ) or not tool_answer \
                                                          else f"{tool_answer}\n" 
                except  Exception  :
                       thought_action_responses[thought] +=  ''     #f"{act}\n"  
        print(f"\033[35mObservation:\n{[x for x in  thought_action_responses.values()]}\033[0m")            

        history = self.format_solver(thought_action_responses,qwen)      
        restore_histoty = []
        for sub_task , tool_answer in history:
            if  tool_answer:
                restore_histoty.append({'user': sub_task, 'assistant': tool_answer})  
        return   restore_histoty








if __name__ == '__main__':           
    from config.parser import args
    from LLM.Qwen import Qwen  
    from tools.call_plugin import User_defined_tools
    from momery.vector_store import task_map_tool
    TOOL = User_defined_tools(args.output_file) 
    qwen = Qwen(args.checkpoint)
    query =  "已知三角形的三边长度分别为15、16、17,计算三角形的外接圆面积？"
    tptu  = TPTU()
    history = tptu( task = query,
          TOOL = TOOL,
          LLM = qwen,
        #   tool_query_func = task_map_tool  # 测试一下效果不好，不建议用
          )
    print(history)

  


    





