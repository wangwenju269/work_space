import json5
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from .tool_desc import TOOLS
from config.parser import DataArguments

class   GET_TOOL_DESC:
        def __init__(self) -> None:
            self.all_tool = TOOLS
            self.TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""
        
        def get_tool_desc(self,list_of_plugin_info) -> str:
            tools_text = []
            for plugin_info in list_of_plugin_info:
                tool = self.TOOL_DESC.format(
                    name_for_model=plugin_info["name_for_model"],
                    name_for_human=plugin_info["name_for_human"],
                    description_for_model=plugin_info["description_for_model"],
                    parameters=json5.dumps(plugin_info["parameters"], ensure_ascii=False),
                )
                if plugin_info.get('args_format', 'json') == 'json':
                    tool += " Format the arguments as a JSON object."
                elif plugin_info['args_format'] == 'code':
                    tool += ' Enclose the code within triple backticks (`) at the beginning and end of the code.'
                else:
                    raise NotImplementedError
                tools_text.append(tool)
            tools_text = '\n\n'.join(tools_text)
            tools_name_text = ', '.join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])
            return tools_text,  tools_name_text
         
    
        '''选用用户指定的工具, 指定格式;(工具1||工具2||工具3||...)'''
        def assign_name_tool(self,name):
            sub_tool = [ plugin for plugin in self.all_tool if  plugin['name_for_model'] in name.split('||') ] 
            return sub_tool
    
        '''根据用户的query的语义信息去检索工具'''
        def model_select_tool(self,  query, k = 4 ):
            '''缓存工具的向量存储器'''
            vector_store = FAISS.from_documents(
                [Document(page_content=plugin['description_for_model'], 
                        metadata={"plugin_name": plugin['name_for_model']}
                        )  for plugin in self.all_tool ],
                ModelScopeEmbeddings(model_id = DataArguments().model_id)
                )
            docs = vector_store.similarity_search(query, k, score_threshold = 100)
            tool_name = [tool.metadata['plugin_name'] for tool in docs]
            sub_tools = [ plugin for plugin in TOOLS if  plugin['name_for_model'] in tool_name ]    
            return sub_tools
        
        
        def tool_set(self,select_tool_task,assign_tool):
            select_tool, task  = select_tool_task
            if  not select_tool and not assign_tool:               # 使用所有的工具集合
                tool_sets = self.all_tool 
            elif assign_tool:
                tool_sets = self.assign_name_tool(select_tool)     # 使用用户指定的工具
            else:    
                tool_sets = self.model_select_tool(task)           # 使用模型选择检索工具
            return tool_sets  