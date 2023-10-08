import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings
warnings.filterwarnings("ignore")
from config.parser import args
from LLM.Qwen import Qwen
from tools.call_plugin import User_defined_tools
from momery.vector_store import task_map_tool
from utils.logger import logger
from pprint import pprint
""" global to load model """
qwen = Qwen(args.checkpoint)
""" Prepare the plug-in tool  """
TOOL = User_defined_tools(args.output_file)

def pipeline(query,history):
    from plan.subplan_react import REACT
    react = REACT()
    response = react(task = query,
                    history = history,
                    LLM = qwen,
                    TOOL = TOOL,
                    tool_query_func = task_map_tool
                    )
    return  response
    

def main(task):
    print('task_question:' ,"\033[31m " + task + " \033[0m ")
    """ 先将复杂问题进行拆解 """
    history = [] 
    if  args.split_subtask:
        from  plan.subplan_tptu import TPTU
        tptu  = TPTU()
        history = tptu( task = task,
                        LLM =  qwen,
                        TOOL = TOOL,
                        # tool_query_func = task_map_tool 
                        )    
    print('multi_chat:') ; pprint(history,indent = 8 ,sort_dicts=False)    
    response = pipeline(task,history)
    return response


if __name__ == "__main__":
    print('\n------------------------------------------')                    
                                                  
    # task = "搜索《咏鹅》古诗的内容信息，绘制一幅包含诗意的图片"                                                            
    task = "请问天津比北京的气温高多少摄氏度？"                                     
    task = "华为首席执行官比百度总裁的年龄大多少岁？"   
    task = "莱昂纳多·迪卡普里奥的女朋友是谁? What is her current age raised to the 0.43 power?"
    task = "解方程 2x + 5 = -3x + 7"   
    task = "使用Math工具解方程 2x + 5 = -3x + 7" 
    task = "论文编号为1605.08386的文章提取关键词"  
    task =  "使用python计算问题,已知三角形的三边长度分别为15、16、17,计算三角形的外接圆面积？" 
    task =  "已知三角形的三边长度分别为15、16、17,计算三角形的外接圆面积？" 
    task = "理解《出师表》的核心思想信息,朗读第一自然段文章内容"   
    task = '给我画个范冰冰明星的头像吧'   
    main(task)
    

 






        # from plan.subplan_Rewoo import Sub_task_Rewoo
        # sub_task_rewoo = Sub_task_Rewoo()
        # history = sub_task_rewoo( task = task,
        #                           LLM  = qwen,
        #                           TOOL = TOOL,
        #                           tool_query_func = task_map_tool )