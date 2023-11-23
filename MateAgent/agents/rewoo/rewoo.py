from .prompt import  PLANNER_PROMPT_CN, SOLVER_PROMPT_CN , WORKER_PROMPT_CN
from utils.build import GET_TOOL_DESC
import re, json5

class  Rewoo(GET_TOOL_DESC):
    def __init__(self,llm, External_API) :
        super().__init__()
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\nYou are a helpful assistant.{self.im_end}'
        self.llm = llm
        self.External_API = External_API

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
        thoughts, action_units = [] , [] 
        thoughts = re.findall('subtasks: (.+)', message)
        action_units = re.findall('#E\[\d\]\s=\s(.+)', message) 
        return thoughts, action_units
    

    @staticmethod
    def parse_paremater(inputs:str):
        '''
        功能：将输入解析为函数名，函数参数形式
        return : 函数名，函数参数
        ''' 
        act_name, variable, input_paremater =  re.findall(r'(.*?)\((.*)="(.*)"\)',inputs.strip())[0]
        input_paremater = json5.dumps({variable:input_paremater},ensure_ascii=False)
        return act_name, input_paremater


    @staticmethod
    def post_process(tool_answer):
        '''
        功能：对工具的解析的结果，进行后处理操作，后续补充。
        '''
        stop_list = ["Wolfram Alpha wasn't able to answer it",'SyntaxError','ModuleNotFoundError','NameError']
        for stop_word in stop_list:
            if stop_word in tool_answer:
               raise Exception("Error calling tool API")
        return str(tool_answer) if tool_answer else '' 
    
    
    def sub_task_process(self,thoughts, actions):
        '''
        01 功能：子任务的处理模块,输入thoughts, actions均为LIST,且元素具有对应关系。
        '''
        history = []
        resources = {}
        for i, (thought, action) in enumerate(zip(thoughts, actions)): 
            act_name,input_paremater = self.parse_paremater(action) 
            for key, values in resources.items():
                if  key in input_paremater:  
                    input_paremater = input_paremater.replace(key,values)  
            observation = self.External_API.call_plugin(act_name,input_paremater)
            print(observation)
            observation = self.post_process(observation)
            resources[f'#E[{i+1}]'] = observation   # 把结果 保存在 资源里 
            history.append((thought,observation))
        return history        


    def Summarize(self,thought,action_return):
        solver_prompt = SOLVER_PROMPT_CN.format(question = thought, worker_log = action_return)
        answer = self.llm.generate(solver_prompt)
        return answer
    
    def conclusion(self,query, history):
        worker_log = ''
        for thought, action_return in history:
            worker_response = WORKER_PROMPT_CN.format(
                              thought=thought,
                              action_resp = action_return)
            worker_log += worker_response
        return self.Summarize(query, worker_log)


    def infer(self,prompt):
        turn_id = 0
        while turn_id < 2:
            response = self.llm.text_completion(prompt, stop_words = ['```'] )     
            response = response.split('```')[0]
            print(response)
            try:  
                   thoughts, actions = self.parse_output(response)
                   break
            except Exception :
                   turn_id += 1
        return thoughts, actions   


    def run(self,task, **kwargs) :
        history = kwargs.pop('history',[])
        select_tool = kwargs.pop('llm_select_tool', False)
        assign_tool = kwargs.pop('user_assign_tool',[])
        tools = self.tool_set((select_tool,task),assign_tool)
        tools_desc, tools_name = self.get_tool_desc(tools) 
        prompt  =  self.construct_prompt(task, tools_desc)
        "模型推理"
        thoughts, actions = self.infer(prompt)
        "子任务处理"
        history += self.sub_task_process(thoughts, actions)
        "总结答复"
        finally_answer = self.conclusion(task, history)
        return finally_answer
  




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    from config.parser import DataArguments
    from llm.model import Qwen  
    from tools.plugins import User_defined_tools
    args = DataArguments()
    External_API = User_defined_tools() 
    qwen = Qwen(args.checkpoint)
    agent  = Rewoo(llm = qwen,
                   External_API = External_API
    )

    agent.run(task = '计算百度总裁年龄的五次幂减去华为首席执行官年龄的三次方,该数值是多少?')

  


    





