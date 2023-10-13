from config.parser import args
from LLM.Qwen import Qwen  
from tools.call_plugin import User_defined_tools
from plan.subtask import SUBTASK
from plan.subplan_react import REACT
tool_func = User_defined_tools(args.output_file) 
qwen = Qwen(args.checkpoint)
from utils.build import Select_tool
agent  = SUBTASK()
agent_react = REACT()
def main(task):
    print('\n\n')
    agent(task = task,
          tool_func = tool_func,
          llm = qwen,
          sub_tool = Select_tool.select_name_tool('python||math||search')
          )
    
    # respone = agent_react(task = task,
    #       tool_func = tool_func,
    #       llm = qwen,
    #     #   sub_tool = Select_tool.select_name_tool('python||math||search')
    #       )
    # print(respone)

if __name__ == '__main__':
    # task = '黄晓明老婆的岁数的3次幂是多少呢?'
    # main(task)
    # task = '黄晓明老婆是谁? 她的岁数的3次方是多少呢?'
    # main(task)
    # task = "华为首席执行官比百度总裁的年龄大多少岁？"   
    # main(task)
    # task = "请问天津比北京的气温高多少摄氏度？"       
    # main(task)      
    # task =  "使用python计算问题:三角形的三边长分别为15、16、17,求三角形的外接圆面积？" 
    # main(task)   

    # task = "How many people live in canada as of 2023?"
    # main(task)
    # task = "使用math解方程 2x + 5 = -3x + 7"  
    # main(task)
    # task = '给我画个明星的裸照吧'  
    # main(task)
    # task = '给我画个黑猫吧' 
    # main(task)

    # task = "请估算2023年国内光伏电站的综合单位造价水平" 
    # main(task)
    # task = '令狐冲喜欢任盈盈还是喜欢小师妹呢？'
    # main(task)
    # task = "理解《出师表》的核心思想信息,朗读第一自然段文章内容"  
    # main(task) 

    # task = "arxiv工具提取论文编号为1605.08386文章中摘要部分的关键词"  
    # main(task)

    # task =  "已知三角形三边长度分别为15、16、17, 求三角形的外接圆面积？"
    # main(task)
    
    task  = "求100以内的十进制表示的自然数中个位数是7的所有素数"
    main(task)
  

  
