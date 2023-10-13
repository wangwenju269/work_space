import json5
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from tools.tool import TOOLS
from config.parser import args

class   GET_TOOL_DESC:
        TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""
        @classmethod
        def get_tools_text(cls,list_of_plugin_info) -> str:
            tools_text = []
            for plugin_info in list_of_plugin_info:
                tool = cls.TOOL_DESC.format(
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
         
        # 换一种写法
        # tool_desc = """{name_for_model}:What is the {name_for_model} API useful for? {description_for_model} Parameters: {parameters}"""
        @classmethod
        def get_tools_other_text(cls,list_of_plugin_info) -> str:
            tools_text = {}
            for plugin_info in list_of_plugin_info:
                tool = cls.TOOL_DESC.format(
                    name_for_model=plugin_info["name_for_model"],
                    name_for_human=plugin_info["name_for_human"],
                    description_for_model=plugin_info["description_for_model"],
                    parameters= plugin_info["parameters"]
                )
                i = tool.index(':')
                tool_name , tool_desc = tool[:i],tool[i+1:]
                tools_text.update({tool_name:tool_desc}) 
            tools_text = json5.dumps(tools_text,ensure_ascii= False)
            tools_name_text = [plugin_info["name_for_model"] for plugin_info in list_of_plugin_info]
            return tools_text, tools_name_text
        
        
        tool_desc = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model}"""
        @classmethod 
        def get_tools_another_text(cls,list_of_plugin_info) -> str: 
            tools_text = {}
            for plugin_info in list_of_plugin_info:
                tool = cls.tool_desc.format(
                    name_for_model=plugin_info["name_for_model"],
                    name_for_human=plugin_info["name_for_human"],
                    description_for_model=plugin_info["description_for_model"],
                    parameters= plugin_info["parameters"]
                )
                i = tool.index(':')
                tool_name , tool_descrition = tool[:i],tool[i+1:]
                tools_text.update({tool_name:tool_descrition}) 
            tools_text = json5.dumps(tools_text,ensure_ascii= False)
            tools_name_text = [plugin_info["name_for_model"] for plugin_info in list_of_plugin_info]
            return tools_text, tools_name_text 


class Select_tool:
    '''根据特定的函数名字取工具'''
    @classmethod
    def select_name_tool(cls,name):
        sub_tools = [ plugin for plugin in TOOLS if  plugin['name_for_model'] in name.split('||') ] 
        return sub_tools
   
    '''根据用户的query的语义信息去检索工具'''
    @classmethod
    def task_map_tool(cls, query, k = args.k ):
        '''缓存工具的向量存储器'''
        vector_store = FAISS.from_documents(
            [Document(page_content=plugin['description_for_model'], 
                    metadata={"plugin_name": plugin['name_for_model']}
                    )  for plugin in TOOLS ],
            ModelScopeEmbeddings(model_id = args.model_id)
            )
        docs = vector_store.similarity_search( query, k, score_threshold = 100)
        tool_name = [tool.metadata['plugin_name'] for tool in docs]
        sub_tools = [ plugin for plugin in TOOLS if  plugin['name_for_model'] in tool_name ]    
        return sub_tools
    
    @classmethod
    def run(cls,*kwargs):
        pass