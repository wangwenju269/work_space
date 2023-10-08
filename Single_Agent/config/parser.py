"""加载环境变量"""
from dotenv import load_dotenv, find_dotenv
load_dotenv()

"""配置模型参数和数据参数"""
from dataclasses import dataclass, field
@dataclass
class ModelArguments:
      checkpoint : str = field(
      default = "/data/lly/model/Qwen-7B-Chat",
      metadata = {"help" : '阿里千问大模型的权重文件'}
      )
      model_id : str = field(
      default = "/data/wangwenju/workspace/damo/nlp_corom_sentence-embedding_chinese-base",
      metadata = {"help" : '小模型用于获取工具描述的嵌入向量'}
      )

@dataclass
class DataArguments(ModelArguments):
      output_file:str = field(
      default = "temp/speech.mp3",
      metadata = {"help" : 'output file'}
      )
      k: int = field(
      default = 2 ,
      metadata = {"help" : "候选工具的数量"}
      )      
      im_start: str = field(
      default = '<|im_start|>' ,
      metadata = {"help" : '<|im_start|>'}
      )   
      im_end: str = field(
      default = '<|im_end|>' ,
      metadata = {"help" :'<|im_end|>'}
      )   
      store_vec: str = field(
      default = 'agent/momery/' ,
      metadata = {"help" :'外部向量FIASS库'}
      )      
      split_subtask: bool = field(
      default = True ,
      metadata = {"help" :'是否需要任务分解'}
      )   


args = DataArguments()



# import argparse
# parser = argparse.ArgumentParser(description="agent")
# parser.add_argument("--checkpoint", type=str, default = "/data/lly/model/Qwen-7B-Chat" ,help="checkpoint file")
# parser.add_argument("--output_file", type=str, default = "temp/speech.mp3" ,help="output file")
# parser.add_argument("--model_id", type=str, default = "/data/wangwenju/workspace/damo/nlp_corom_sentence-embedding_chinese-base" ,help="emb")
# parser.add_argument("--k", type = int, default = 3 ,help="候选工具的数量")
# args = parser.parse_args()