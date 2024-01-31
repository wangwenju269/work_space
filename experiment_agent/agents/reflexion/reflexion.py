from .prompt import REFLEXION_REACT_PROMPT
from .react import React
class Reflexion(React):
    def __init__(self,
                llm,
                External_API
                ):
        super(Reflexion,self).__init__(llm,External_API)  
        self.prefix = f'{self.im_start}system\nYou are a reflect assistant.{self.im_end}'

    def reset(self):    
        self.reflections = []
        self.reason_path = ''    # 记录推理路线
        self.reflect_path = ''   # 记录反思路线
        self.short_money = ''    # 短期记忆
        self.Last_round_response = ''  # 记录上一轮的回应
    

    def construct_reflect_prompt(self, query,tool_descs):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.prefix 
        query = REFLEXION_REACT_PROMPT.format(
                tool_descs = tool_descs,
                query = query )
        query = query.lstrip('\n').rstrip()              # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt 
    

    def infer(self,task,history, planning_prompt, tools_desc, tools_name):
        # text = ''
        count = 0 
        while count <= 5:
            response = self.llm.text_completion(planning_prompt, stop_words=['Observation:', 'Observation:\n'])
            action, action_input, _ = self.parse_latest_plugin_call(response)
            if  (action in tools_name) and action :  # 需要调用插件
                try:
                    api_output = self.External_API.call_plugin(action.lower(), action_input)
                    api_output = self.post_process(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
                    if "no tool founds" == api_output:
                        break
                    print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
                    planning_prompt +=  f'{response} {api_output}\n<|im_end|>\n<|im_start|>assistant\n'
                    self.reason_path += f'{response} {api_output}\n'
                    self.short_money += f'{response} {api_output}\n'
                except:
                    self.reason_path +=  f'{response} Answer is INCORRECT\n' 
                    self.short_money +=  f'{response} Answer is INCORRECT'  
                    """ 反思 过程 """
                    # response = response.replace('Observation:','')
                    # response = response.lstrip('\n').rstrip() 
                    self.reflect_path += self.short_money
                    reflect_prompt = self.construct_reflect_prompt(f'{task}\n{self.reflect_path}', tools_desc)
                    reflect_response = self.llm.text_completion(reflect_prompt,stop_words=['Thought','Thought:\n',
                                                                                           'Action', 'Action:\n'
                                                                                           'Observation','Observation:\n'
                                                                                           ])
                    reflect_response = reflect_response.replace('Begin!','')  \
                                                       .replace('Thought','').replace('Action','').replace('Observation','') \
                                                       .lstrip('\n').strip('\n').rstrip().strip()
                    print("\033[32m" + response + "\033[0m" + "\033[31m" + '\nReflection:' + reflect_response + "\033[0m")
                    
                    '''将 反思的结果 加入 self.reflect_path'''
                    self.reflections.append(reflect_response)
                    '''
                    # 方法1：将反思的过程加入 反思思考链 : self.reflect_path += f'\nReflection:{reflect_response}\n'
                    # 方法2：反思链式不记录该轮的信息 :  self.reflect_path = self.reflect_path[: -len(f'{response} Answer is INCORRECT')]
                    '''
                    self.reflect_path = self.reflect_path[: -len(f'{response} Answer is INCORRECT')]
                    """ 重置 推理 环境 """ 
                    planning_prompt  =  self.construct_react_prompt( 
                                                query = f'{task}\n{self.reason_path}', 
                                                history = history,
                                                tool_desc = tools_desc,
                                                tool_name = tools_name,
                                                reflection = self.reflections,
                                                )
                    """短期动态记忆空间清空"""
                    self.short_money = ''    
            else: 
                break
            count += 1
        print(f"\033[32m{response}\033[0m")
        self.reason_path += f'\n{response}'
        return response
    
    def run(self,task, **kwargs) :
        history = kwargs.pop('history',[])
        select_tool = kwargs.pop('llm_select_tool', False)
        assign_tool = kwargs.pop('user_assign_tool',[])
        tools = self.tool_set((select_tool,task),assign_tool)
        print(f"Origin Query:{task}")
        tools_desc, tools_name = self.get_tool_desc(tools) 
        planning_prompt  =  self.construct_react_prompt( 
                                                query = task, 
                                                history = history,
                                                tool_desc = tools_desc,
                                                tool_name = tools_name
                                                )
        '''------  获取模型回应 -------'''
        response = self.infer(task,history,planning_prompt, tools_desc, tools_name) 
        response = response.lstrip('\n').rstrip() 
        return response


if __name__ == '__main__':
    from config.parser import DataArguments
    from llm.model import Qwen
    from tools.plugins import User_defined_tools
    External_API = User_defined_tools() 
    args = DataArguments()
    qwen = Qwen(args.checkpoint)
    react = Reflexion(qwen,External_API)  
    task = '计算百度总裁年龄的五次幂减去华为首席执行官年龄的三次方,该数值是多少?'     
    react.run(task)