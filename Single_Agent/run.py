from config.parser import args
from LLM.Qwen import Qwen  
from tools.call_plugin import User_defined_tools
from plan.subtask import SUBTASK
from plan.subplan_react import REACT
from utils.build import Select_tool

External_API = User_defined_tools() 
qwen = Qwen(args.checkpoint)
agent  = SUBTASK()
agentx = REACT()
def main(task):
    print('\nTPTU:\n')
    agent(task = task,
          External_API = External_API,
          llm = qwen,
          # user_assign_tool = Select_tool.assign_name_tool('python||math||search')
          )
    # print('\nreact:\n')
    # agentx(task = task,
    #       External_API = External_API,
    #       llm = qwen,
    #       # user_assign_tool = Select_tool.assign_name_tool('python||math||search')
    #       )



if __name__ == '__main__':
    
    # task = "请问天津与北京的气温相差多少摄氏度？"       
    # main(task)       
    # task = "请问天津比北京的气温高(低)多少摄氏度？"       
    # main(task) 
    # task = "华为首席执行官比百度总裁的年龄大多少岁？"   
    # main(task)
    # task = '黄晓明老婆的岁数的3次方是多少呢?'
    # main(task)

    task = '朗读一下苏轼的《定风波》这首词的完整内容'
    main(task)
    # task = '直接朗读一下《蜀道难》这首文章的全部段落'
    # main(task)
    # task = "解方程 2x + 5 = -3x + 7"  
    # main(task)
    # task = "提取arxiv论文编号为1605.08386文章摘要部分的关键词"  
    # main(task)

    # task = "请估算2023年国内光伏电站的综合单位造价水平" 
    # main(task)
    # task = '令狐冲喜欢任盈盈还是喜欢小师妹呢？'
    # main(task)


    # task = '给我画个黑猫吧' 
    # main(task)
    # task = '给我画个建筑施工现场的危险场景的图片吧' 
    # main(task)
    # task = '给我画个明星的裸照吧'  
    # main(task)
    # task =  "使用python计算问题:三角形的三边长分别为15、16、17,求三角形的外接圆面积？" 
    # main(task)  



    # task = '黄晓明老婆是谁? 她的岁数的3次幂是多少呢? Provide me with the problem PDDL file that describes the planning problem directly without further explanations.'
    # main(task)
    # task = '计算百度总裁年龄的三次幂减去华为首席执行官年龄的平方,该数值是多少?'
    # main(task)
    # task =  "81728738271872871871672 + 28781729836746721"
    # main(task)
    # task  = "求100以内的十进制表示的自然数中个位数是7的所有素数"
    # main(task)


