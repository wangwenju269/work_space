import sys
sys.path.append('wangGPT')
from prompt.analysis_prompt import PLANNER_PROMPT_CN, EXAMPLES
class Rewoo:
    def __init__(self) :
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\nYou are a task decomposer.{self.im_end}'

    def construct_prompt(self,query):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.prefix
        query = PLANNER_PROMPT_CN.format(question = query, EXAMPLES = EXAMPLES)
        query = query.lstrip('\n').rstrip()   
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt
    

    def parser_output(self,text):
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

    def run(self, task, LLM):
        prompt_rewoo = self.construct_prompt(task)
        response = LLM.text_completion(prompt_rewoo, stop_words=['<|im_end|>'])
        sub_task = self.parser_output(response)  
        return sub_task
       

if __name__ == '__main__':           
    from config.parser import args
    from LLM.Qwen import Qwen  
    qwen = Qwen(args.checkpoint)
    agent = Rewoo()

    query = "黄晓明老婆是谁? 她的岁数的3次幂方是多少呢?"  
    respose = agent.run(query,qwen)
    print(respose)

    task = '计算百度总裁年龄的三次幂减去华为首席执行官年龄的平方,该数值是多少?'
    respose = agent.run(task,qwen)
    print(respose)


    task = "请问天津与北京的气温相差多少摄氏度？"       
    respose = agent.run(task,qwen)
    print(respose)   
    task = "请问天津比北京的气温高(低)多少摄氏度？"       
    respose = agent.run(task,qwen)
    print(respose)
    task = "华为首席执行官比百度总裁的年龄大多少岁？"   
    respose = agent.run(task,qwen)
    print(respose)
    task = '黄晓明老婆的岁数的3次方是多少呢?'
    respose = agent.run(task,qwen)
    print(respose)



    task = '朗读一下《出师表》这首文章的第一段落'
    respose = agent.run(task,qwen)
    print(respose)
    task = '朗读一下《琵琶行》这首文章的第一段落'
    respose = agent.run(task,qwen)
    print(respose)
    task = "解方程 2x + 5 = -3x + 7"  
    respose = agent.run(task,qwen)
    print(respose)
    task = "提取arxiv论文编号为1605.08386文章摘要部分的关键词"  
    respose = agent.run(task,qwen)
    print(respose)

    task = "请估算2023年国内光伏电站的综合单位造价水平" 
    respose = agent.run(task,qwen)
    print(respose)
    task = '令狐冲喜欢任盈盈还是喜欢小师妹呢？'
    respose = agent.run(task,qwen)
    print(respose)


    task = '给我画个黑猫吧' 
    respose = agent.run(task,qwen)
    print(respose)
    task = '给我画个建筑施工现场的危险场景的图片吧' 
    respose = agent.run(task,qwen)
    print(respose)
    task = '给我画个明星的裸照吧'  
    respose = agent.run(task,qwen)
    print(respose)
    task =  "使用python计算问题:三角形的三边长分别为15、16、17,求三角形的外接圆面积？" 
    respose = agent.run(task,qwen)
    print(respose)



    task = '黄晓明老婆是谁? 她的岁数的3次幂是多少呢?'
    respose = agent.run(task,qwen)
    print(respose)
    task = '计算百度总裁年龄的三次幂减去华为首席执行官年龄的平方,该数值是多少?'
    respose = agent.run(task,qwen)
    print(respose)
    task =  "已知三角形三边长度分别为15、16、17, 求三角形的外接圆面积？"
    respose = agent.run(task,qwen)
    print(respose)
    task  = "使用python计算问题:求100以内的十进制表示的自然数中个位数是7的所有素数"
    respose = agent.run(task,qwen)
    print(respose)