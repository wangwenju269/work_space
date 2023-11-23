"""加载环境变量"""
from dotenv import load_dotenv, find_dotenv
load_dotenv()

"""配置模型参数和数据参数"""
from dataclasses import dataclass, field
@dataclass
class ModelArguments:
      checkpoint : str = field(
      default = "/data/kongsy-a/basemodels/Qwen-7B-Chat", # "/data/lly/model/Qwen-7B-Chat",
      metadata = {"help" : '阿里千问大模型的权重文件'}
      )


@dataclass
class DataArguments(ModelArguments):
      output_vidio_file:str = field(
      default = "./MateAgent/assets/vidio",
      metadata = {"help" : 'output file'}
      )

      output_speech_file:str = field(
      default = "./MateAgent/assets/speech",
      metadata = {"help" : 'output file'}
      ) 

      output_image_file:str = field(
      default = "./MateAgent/assets/images",
      metadata = {"help" : 'output file'}
      )

      k: int = field(
      default = 4 ,
      metadata = {"help" : "based on query to select k-number of tools"}
      )      

      split_subtask: bool = field(
      default = True ,
      metadata = {"help" :'subtask'}
      )   
        
        




