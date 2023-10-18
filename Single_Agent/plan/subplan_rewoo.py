import sys
sys.path.append('agent')
from prompt.rewoo import PLANNER_PROMPT_CN , EXAMPLES, CALL_PROTOCOL_CN
class Input_prifix:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\nYou are a task decomposer.{self.im_end}'

class  Rewoo:
    @classmethod 
    def construct_prompt(cls,query):
        prefix =  Input_prifix()
        im_start, im_end, prompt = prefix.im_start,  prefix.im_end, prefix.prefix
        query = PLANNER_PROMPT_CN.format(question = query, EXAMPLES = EXAMPLES)
        query = query.lstrip('\n').rstrip()   
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt
    
    @classmethod
    def parser_output(cls,text):
        import re
        text = text.split('```')[0]
        text = text     if text.endswith('\n')  else text + '\n'
        "正则匹配的方法去解析"
        try:
           pattern = r"(Complex issue|复杂问题|Subtask):\s?(.*?)\n"
           matches = re.findall(pattern, text)
           flag = matches[0][1].rstrip()
           sub_task = [x[1].rstrip() for x in matches[1:]]
           return sub_task if flag == 'Yes' else []
        except:
            print('----正则解析失败----')     
        return 

class Sub_task_Rewoo:
    def sub_task_func(self,task,LLM, **kwargs):
        prompt_rewoo = Rewoo.construct_prompt(task)
        response = LLM.text_completion(prompt_rewoo, stop_words=['<|im_end|>'])
        sub_task = Rewoo.parser_output(response)  
        from tools.tool import tools   
        sub_tool_names = [[tools[i]['name_for_model'] for i in range(len(tools))] for _ in sub_task]

        return sub_task, sub_tool_names
    
    def __call__(self, task,  LLM, tool_func, **kwargs):
        sub_task, sub_tool_names = self.sub_task_func(task, LLM, **kwargs)
        print('split_sub_task:\n',"\033[33m " + str(sub_task) +  "\033[0m ")
        print('match_sub_tool:\n',"\033[36m " + str(sub_tool_names) +  "\033[0m ") 
        history, restore_histoty, new_history = [] , [], [] 
        for  query , tool_name in zip(sub_task, sub_tool_names):
            inputs = tool_func._construct_Input(query,tool_name[0])
            func = tool_func.config[tool_name[0]]
            tool_answer = func(inputs)
            if  not tool_answer:
                print( "\033[33mQ:" + query + "\033[0m\n" +
                   "\033[34mA:" + '文件保存在：/workspace/temp' + "\033[0m" )
                continue
            if 'image_url' in tool_answer:
                restore_histoty.append({'user': query, 'assistant': tool_answer})  
                continue
            prompts = CALL_PROTOCOL_CN.format(
                        origin_question = task,
                        reference_information = tool_answer,
                        sub_question = query
                        )  
            answer, _ = LLM.model.chat(
                                tokenizer = LLM.tokenizer,
                                query = prompts,
                                history = history,
                                append_history = False)  
            new_history.append((query,answer))
            history = new_history   
            restore_histoty.append({'user': query, 'assistant': answer})   
            print( "\033[33mQ:" + query + "\033[0m\n" +
                   "\033[34mA:" + answer + "\033[0m" )
        return restore_histoty

if __name__ == '__main__':           
    from config.parser import args
    from LLM.Qwen import Qwen  
    from tools.call_plugin import User_defined_tools
    qwen = Qwen(args.checkpoint)
    tool_func = User_defined_tools() 
    query = "黄晓明老婆是谁? 她的岁数的3次幂方是多少呢?"  
    respose = Sub_task_Rewoo(query,qwen,tool_func)
    print(respose)

