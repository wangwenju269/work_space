import os, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import warnings
warnings.filterwarnings("ignore")
from config.parser import DataArguments
from llm.model import Qwen  
from tools.plugins import User_defined_tools
from agents import Reflexion,React,Rewoo
import json
def del_env_path(file_path_folder):
    ls = os.listdir(file_path_folder)
    for s in ls :
        f_path = os.path.join(file_path_folder, s)
        if os.path.isdir(f_path):
           del_env_path(f_path) 
        else:   
           os.remove(f_path)

def upload_env_path(args):
    test_folder = args.test_data
    ls = os.listdir(test_folder)
    for file_path in ls:
        f_path = os.path.join(test_folder, file_path)
        file_names = os.path.basename(file_path)
        _ , suffix = file_names.split('.')
        if suffix in ['pdf']:
            shutil.move(f_path, args.pdf_file)
        elif suffix  in ['png','jpg']:
            shutil.move(f_path, args.image_file)
        else:
            raise  ValueError("{0} format is not supported, Only supports[/.pdf/.jpg/.png]formats ".format(suffix))
        
def dict2json(file_name,the_dict):
    '''
    将字典文件写如到json文件中
    :param file_name: 要写入的json文件名(需要有.json后缀),str类型
    :param the_dict: 要写入的数据，dict类型
    :return: 1代表写入成功,0代表写入失败
    '''
    try:
        json_str = json.dumps(the_dict,indent=4,ensure_ascii=False)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0

args = DataArguments()
qwen = Qwen(args.checkpoint)
External_API =  User_defined_tools(llm = qwen,
                                  args = args
                                  ) 
react_agent =  React(   External_API = External_API,
                        llm = qwen
                    )

def excutor(task):
    print('\n\n\n')
    # 收集用户的id和体验用户数据信息
    import uuid
    unique_id = uuid.uuid4()
    react_agent.unique_id = str(unique_id)

    from collections import defaultdict
    react_agent.collect_user_data = defaultdict(list)
    react_agent.collect_user_data[react_agent.unique_id].extend(
            [{'from':"<|system|>",
            'value':"<|im_start|>system\n你有多种能力，可以通过插件集成api来回复用户的问题，还能解答用户使用模型遇到的问题和模型知识相关问答。目前支持的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。<|im_end|>\n" }
            ])

    """ 清空环境里文件信息"""  
    # del_env_path(args.file_folder)
    """ 上传数据 """
    # upload_env_path(args) 
    react_agent.run(task,
                    user_assign_tool = 'pdf_parser||table_cell_extract||math||safety_belt_checker||text_speech'
                    )
    # """ 记得清空环境里文件信息"""
    # del_env_path(args.file_folder)
    '收集到的用户数据进行写出'
    dict2json(f'/data/wangwj-t/workspace/experiment/collect_user_data/{unique_id}.json', react_agent.collect_user_data) 


task = '先检测<安全带.jpg>是否存在风险隐患,后获取<燃气工程.pdf>文件中的综合管线横断面设计图车行道宽度是多少米？已知每米收费5.43元,计算总费用,将结果用语音回复'
# task = '先检测<安全带.jpg>是否存在风险隐患,后获取<燃气工程.pdf>文件中的综合管线横断面设计图车行道宽度是多少米？已知每米收费5.43元,计算总费用,将结果用语音回复。最后,提取燃气管道纵断面设计图表格里的设计总负责人是谁'
excutor(task)


#  cp -r experiment/data/*  experiment/test_data   
# text = """Your task is to rewrite the input query into a query that makes it easier to answer this query,
# and you are required to give the thought process and then the query rewriting result. 
# The thought process and the query rewriting result are separated by a semicolon.
# Query: 先检测<安全带.jpg>是否存在风险隐患,后获取<燃气工程.pdf>文件中的综合管线横断面设计图车行道宽度是多少米？已知每米收费5.43元,计算总费用,将结果用语音回复。最后,提取燃气管道纵断面设计图表格里的设计总负责人是谁
# """   
# task = """
# 1. 检查<安全带.jpg>是否存在风险隐患。
# 2. 从<燃气工程.pdf>文件中获取综合管线横断面设计图，车行道宽度是多少米？已知每米收费5.43元，计算总费用，并用语音回复结果。
# 3. 从燃气管道纵断面设计图表格中提取设计总负责人是谁。
# """     
# if __name__ == '__main__':
    # task = "请问天津与北京的气温相差多少摄氏度？"       
    # main(task)     
    # task = "求解方程 2x + 5 = -3x + 7"  
    # main(task)
    # task = '黄晓明老婆的岁数的3次方是多少呢?'
    # main(task)
    # task = '给我画个建筑施工现场的危险场景的图片吧' 
    # main(task)
    # task = '您是谁' 
    # main(task)
    # task = '令狐冲喜欢任盈盈还是喜欢小师妹呢？'
    # main(task)
    # task = 
    # main(task)
    # task = "华为首席执行官比百度总裁的年龄大多少岁？"   
    # main(task)
    # task = "请问广东与哈尔滨的气温相差多少摄氏度？"       
    # main(task) 
    # task = "提取arxiv论文编号为1605.08386文章摘要部分的关键词"  
    # main(task)
    # task = '黄晓明老婆是谁? 她的岁数的3次幂是多少呢?'
    # main(task)
    # task = "请估算2023年国内光伏电站的综合单位造价水平" 
    # main(task)
    # task =  "已知三角形三边长度分别为15、16、17, 求三角形的外接圆面积？"
    # main(task)
    # task  = "使用python计算问题:求100以内的十进制表示的自然数中个位数是7的所有素数"
    # main(task)
    # task =  "使用python计算问题:三角形的三边长分别为15、16、17,求三角形的外接圆面积？" 
    # main(task)  

