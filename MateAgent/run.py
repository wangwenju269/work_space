import sys
sys.path.append('MateAgent')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from config.parser import DataArguments
from llm.model import Qwen  
from tools.plugins import User_defined_tools
from agents import Reflexion


args = DataArguments()
qwen = Qwen(args.checkpoint)
External_API = User_defined_tools(llm  = qwen,
                                  args = args)

react_agent = Reflexion( External_API = External_API,
                         llm = qwen)


task = '您好'
answer = react_agent.run(task)



