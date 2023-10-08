import sys
sys.path.append('agent')
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from tools.tool import tools
from langchain.schema import Document
from config.parser import args

docs = [
        Document(page_content=plugin['description_for_model'], 
                metadata={"plugin_name": plugin['name_for_model']}
                )
        for plugin in tools
    ]

embeddings = ModelScopeEmbeddings(model_id = args.model_id)
vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local(args.store_vec)
# vector_store = FAISS.load_local(folder_path = args.store_vec,
#                                 embeddings = embeddings,
#                                 index_name = 'index' )

def task_map_tool(query, k = args.k ):
    docs = vector_store.similarity_search(query, k )
    tool_name = [tool.metadata['plugin_name'] for tool in docs]
    sub_tools = [ plugin for plugin in tools if  plugin['name_for_model'] in tool_name ]    
    return sub_tools

if  __name__ == '__main__':
    query = "一个新程序在第一个月就有 60 次下载。第二个月的下载量是第一个月的三倍，但第三个月却减少了30%。 三个月内该程序的总下载量是多少"        
    query = "使用Math工具求解方程 2x + 5 = -3x + 7"                         
    task_map_tool(query)

