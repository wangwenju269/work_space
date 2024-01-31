import json5
from .base import Base
import copy
import os
import re
from .image_gen import Image_gen
class User_defined_tools(Base):
        def __init__(self,llm = None , **kwargs):
            super().__init__()
            self.llm = llm
            self.kwargs = kwargs                           # 存放 初始化 资源 信息  
            self.kwargs_copy = copy.deepcopy(kwargs)       # kwargs 不断发生迭代更新，将初始化的参数 另存一份 
            plugins = {
                    'weather_api'   : self.weather_api,
                    'text_speech'      : self.text_speech,
                    'image_gen'        : self.image_gen,
                    'math' : self.math,
                    }
            self.plugins.update(plugins)
        
        def reset(self):
            self.kwargs = self.kwargs_copy

        def math(self,query):
            from .math_api import Math
            if not self.llm : raise  'math 工具缺少 llm 驱动' 
            Math_Api = Math(self.llm) 
            answer = Math_Api.run(query)
            return answer
        

        
        def weather_api(self,query):
            from langchain.utilities import OpenWeatherMapAPIWrapper
            OpenWeatherMap = OpenWeatherMapAPIWrapper()
            query = json5.loads(query)["location"]
            if  all(map(lambda c:'\u4e00' <= c <= '\u9fa5',query)):
                try :    from translate import Translator
                except:  raise ImportError('lack of translate packages')  
                query = Translator(from_lang="Chinese",to_lang="English").translate(query)  
            return OpenWeatherMap.run(query) 
             
        def text_speech(self,action):
            path_file = self.kwargs.get('args').speech_file
            try :   import edge_tts, asyncio, json5
            except:  raise ImportError('lack of [edge_tts, asyncio] packages')  
            text = json5.loads(action)["text"]
            # response = self.llm.generate(f'直接回答：{query}')     # 不够智能，需要进一步做回应
            # if query in response : query = ''
            # text = f'{query}\n{response}'
            voice = 'zh-CN-liaoning-XiaobeiNeural'; rate = '-4%'; volume = '+0%'
            async def my_function():
                tts = edge_tts.Communicate(text = text, voice=voice, rate=rate, volume=volume)
                await tts.save(f'{path_file}/speech.wav') 
            asyncio.run(my_function())   
            return f'我已完成朗读,内容为\n{text}'
        
        def image_gen(self,query):
            path_file = self.kwargs.get('args').image_file
            instances = Image_gen()  
            answer =  instances(query,path_file)  
            return answer  





