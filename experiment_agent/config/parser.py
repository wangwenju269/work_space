"""加载环境变量"""
from dotenv import load_dotenv, find_dotenv
load_dotenv()

"""配置模型参数和数据参数"""
from dataclasses import dataclass, field
@dataclass
class ModelArguments:
      checkpoint : str = field(
      default = "/data/kongsy-a/basemodels/Qwen-14B-Chat", # "/data/lly/model/Qwen-7B-Chat",
      metadata = {"help" : '阿里千问大模型的权重文件'}
      )


@dataclass
class DataArguments(ModelArguments):

      file_folder: str = field(
      default = "./experiment/environment/resource",
      metadata = {"help" : 'input file'}
      )

      test_data: str = field(
      default = "./experiment/test_data",
      metadata = {"help" : 'input file'}
      ) 

      vidio_file:str = field(
      default = "./experiment/environment/resource/vidio",
      metadata = {"help" : 'input file'}
      )

      speech_file:str = field(
      default = "./experiment/environment/resource/speech",
      metadata = {"help" : 'input file'}
      ) 

      image_file:str = field(
      default = "./experiment/environment/resource/images",
      metadata = {"help" : 'input file'}
      )

      pdf_file:str = field(
      default = "./experiment/environment/resource/pdf",
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
        
        




