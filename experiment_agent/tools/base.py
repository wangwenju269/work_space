import json5
from langchain.utilities import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from abc import abstractmethod
search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
python = PythonAstREPLTool()

class Base:
        def __init__(self):
            self.plugins = {
                    'search'  :     self.tool_wrapper(search), 
                    'math'    :     self.tool_wrapper(WolframAlpha),
                    'python'  :     self.tool_wrapper(python),
                    }
            
        @abstractmethod
        def reset(self):
            '重置环境'
            raise NotImplementedError
        
        def tool_wrapper(self, tool):
            def tool_(query):
                query = json5.loads(query)["query"]
                return tool.run(query)
            return tool_
        
        def call_plugin(self, action, action_input):
            func = self.plugins[action]
            observation = func(action_input)
            return  observation 

