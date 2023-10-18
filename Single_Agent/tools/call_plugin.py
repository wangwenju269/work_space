import json5
from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.tools.python.tool import PythonAstREPLTool

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
python = PythonAstREPLTool()
OpenWeatherMap = OpenWeatherMapAPIWrapper()

class User_defined_tools:
        def __init__(self):
            self.config = {
                    'search'  :     self.tool_wrapper_for_qwen(search), 
                    'google_search'  :     self.tool_wrapper_for_qwen(search),
                    # 'quark_Search'   : 
                    'math'    :     self.tool_wrapper_for_qwen(WolframAlpha),
                    'arxiv'   :     self.tool_wrapper_for_qwen(arxiv),
                    'python'  :     self.tool_wrapper_for_qwen(python),
                    'weather_api'   : self.tool_weather_for_qwen(OpenWeatherMap),
                    'speech_synthesis'      : self.text_speech_for_qwen,
                    'image_gen'        : self.text_image_for_qwen,
                    'no_use_tool'      : self.no_use_tool
                    }
            
        def tool_wrapper_for_qwen(self, tool):
            def tool_(query):
                query = json5.loads(query)["query"]
                return tool.run(query)
            return tool_
        
        def tool_weather_for_qwen(self,tool):
            def tool_(query):
                query = json5.loads(query)["location"]
                if  all(map(lambda c:'\u4e00' <= c <= '\u9fa5',query)):
                    from translate import Translator
                    query = Translator(from_lang="Chinese",to_lang="English").translate(query)  
                return tool.run(query)
            return tool_

        def text_speech_for_qwen(self,query):
            import edge_tts, asyncio,json5
            query = json5.loads(query)["prompt"]
            voice = 'zh-CN-YunxiNeural'; rate = '-4%'; volume = '+0%'
            async def my_function():
                tts = edge_tts.Communicate(text = query, voice=voice, rate=rate, volume=volume)
                await tts.save(self.output_file) 
            return asyncio.run(my_function())

        def text_image_for_qwen(self,query):
                import urllib.parse , json5
                prompt = json5.loads(query)["prompt"]
                prompt = urllib.parse.quote(prompt)
                return json5.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
        def no_use_tool(self,x):
            return 'Refer to contextual information to answer questions,plase think it step by step'

        def call_plugin(self, action, action_input):
            func = self.config[action]
            observation = func(action_input)
            return  observation 
        
        @ classmethod
        def _construct_Input(cls,input,tool_name):
            if   tool_name in ['search','google_search','math','arxiv','python' ]:  
                 sub_query = json5.dumps({'query':input},ensure_ascii= False) 
            elif tool_name == 'weather_api':
                 sub_query = json5.dumps({'location':input},ensure_ascii= False) 
            else:
                 sub_query = json5.dumps({'prompt':input},ensure_ascii= False) 
            return  sub_query   