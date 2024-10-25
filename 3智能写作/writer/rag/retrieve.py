import asyncio
from typing import List,Tuple, Union
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from metagpt.rag.schema import BM25RetrieverConfig
from metagpt.ext.writer.actions import RelateFilter,GenKeyword
from metagpt.rag.engines import SimpleEngine
from metagpt.rag.schema import BM25RetrieverConfig, LLMRankerConfig,FAISSIndexConfig,FAISSRetrieverConfig

COARSE_CHUNK_SIZE = 2048
FINE_CHUNK_SIZE = 1024
relate_filter = RelateFilter()
# 定义一个继承自SimpleEngine的类SimpleEngines，用于设置和获取retriever属性
class SimpleEngines(SimpleEngine):
        @property
        def retriever(self):
            return self._retriever

        @retriever.setter
        def retriever(self, value):
            if not isinstance(value, BM25Retriever): 
                raise TypeError("retriever must be an instance of BM25Retriever")
            self._retriever = value

            
def build_engine(ref_dir: Path, persist_dir: Path, model_name: str = "bge-large-zh") -> SimpleEngine:
        retriever_configs = [BM25RetrieverConfig(similarity_top_k= 3)]
        # ranker_configs = [LLMRankerConfig()]

        if persist_dir.exists():
            engine = SimpleEngines.from_index(index_config=FAISSIndexConfig(persist_path=persist_dir),
                                            retriever_configs=retriever_configs
                                            # ranker_configs=ranker_configs
                                            )
        else:
            engine = SimpleEngines.from_docs(input_dir=ref_dir,
                                            retriever_configs=retriever_configs,
                                            # ranker_configs=ranker_configs,
                                            transformations = [SentenceSplitter(chunk_size = COARSE_CHUNK_SIZE)])
            engine.retriever.persist(persist_dir=persist_dir)
        return engine

    
def retrieve_decorator(func):
        async def wrapper(self, *args, **kwargs):
                original_nodes = self.retriever._nodes
                fine_grained_nodes, key_words = await func(original_nodes, *args, **kwargs)
                self.retriever._nodes = []   # 清空检索节点
                self.retriever.add_nodes(fine_grained_nodes)  # 加入粗粒度筛选后节点
                try :
                    nodes = await  self.aretrieve(key_words)  # 执行 BM25 + llm_rank(会解析报错) 
                except:     
                    similarity_top_k = self.retriever._similarity_top_k
                    self.retriever._similarity_top_k = 10
                    nodes = await  self.retriever.aretrieve(key_words) # 执行 BM25 关键词召回
                    self.retriever._similarity_top_k = similarity_top_k
                self.retriever._nodes = original_nodes
                return nodes
        return wrapper



async def doc_ralate_filter(coarse_grained_nodes:Union[List[TextNode], List[str]],
                            title:str,
                            )-> Tuple[Union[List[TextNode],List[str]], List[str]]:
            features = []
            for node in coarse_grained_nodes:
                if isinstance(node,TextNode):
                   node = node.text 
                answer = relate_filter.run(instruction = title, knowledge = node)
                features.append(answer)
            features = await asyncio.gather(*features)
            coarse_filter_nodes, pre_ans = relate_filter.parser_out_filter_nodes(coarse_grained_nodes, features)           
            return  coarse_filter_nodes, pre_ans


@retrieve_decorator
async def key_word_retrieve(coarse_grained_nodes:List[TextNode],
                                title:str,
                                chunk_size:int = FINE_CHUNK_SIZE,
                                max_retry:int = 3
                                ) -> Tuple[List[TextNode], str]:
                # use a large model to screen the coarse-grained relevance of nodes
                coarse_filter_nodes, pre_ans = await doc_ralate_filter(coarse_grained_nodes,title)
                fine_grained_nodes = SentenceSplitter(chunk_size = chunk_size)(coarse_filter_nodes)
                new_nodes = []
                save_org_answer = True
                context = ''.join(pre_ans)             
                while  max_retry and len(context) > chunk_size  :
                        new_node, pre_ans = await doc_ralate_filter(pre_ans, title)
                        if  save_org_answer:
                            for node in new_node: 
                                if  isinstance(node,str):
                                    new_nodes.append(TextNode(text=node, metadata = {'file_path': 'summary_answer'}) )
                                else:
                                    new_nodes.append(node)    
                        context, save_org_answer = ''.join(pre_ans) , False
                        max_retry -= 1
                          
                key_words = await GenKeyword().run(messages = f"{title},{context}" )
                return fine_grained_nodes + new_nodes, key_words        
            