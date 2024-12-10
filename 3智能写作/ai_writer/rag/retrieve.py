import asyncio
from typing import List
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

from metagpt.rag.engines import SimpleEngine
from metagpt.llm import LLM
from metagpt.rag.factories.llm import RAGLLM
from metagpt.ext.writer.actions import RelateFilter
from metagpt.rag.schema import BM25RetrieverConfig, BGERerankConfig,FAISSIndexConfig,FAISSRetrieverConfig


class CustomRetriever(BaseRetriever):
    
    relate_filter: RelateFilter
    
    def __init__(self, nodes: TextNode, relate_filter: RelateFilter = None):
        """Init params."""
        self._nodes = nodes
        self.relate_filter = relate_filter or RelateFilter()
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        features = []
        for node in self._nodes:
            if isinstance(node, TextNode):
                node_text = node.text
            answer = self.relate_filter.run(instruction=query_bundle, knowledge=node_text)
            features.append(answer)
        features = await asyncio.gather(*features)
        filter_nodes, hyde = self.relate_filter.parser_out_filter_nodes(self._nodes, features)
        assert len(filter_nodes) == len(hyde), "Length of filter_nodes and hyde must be equal"
        node_with_scores: List[NodeWithScore] = []
        for node, preliminary_answer in zip(filter_nodes, hyde):
            node.metadata.update({'preliminary_answer': preliminary_answer})
            node_with_scores.append(NodeWithScore(node=node))
        return node_with_scores
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError("The _retrieve method is not implemented.")
    
    
class SimpleEngines(SimpleEngine):
    """
    A subclass of SimpleEngine that provides additional functionality for building and managing engines.
    """

    @property
    def retriever(self):
        
        return self._retriever

    @retriever.setter
    def retriever(self, value):
        
        self._retriever = value

    @classmethod
    def build_advanced_engine(
        cls,
        input_dir: str = None, 
        input_files: List[str] = None,  
        persist_dir: Path = None, 
        **kwargs
    ) -> SimpleEngine:
        """
        Builds an engine instance based on the provided configurations.

        Args:
            input_dir: Directory containing input files.
            input_files: List of input file paths.
            persist_dir: Directory to persist the index.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of SimpleEngine.
        """
          
        embed_model = kwargs.pop('embed_model')
        rerank_model = kwargs.pop('rerank_model')
        chunk_size = kwargs.pop('chunk_size', 512)
        chunk_overlap = kwargs.pop('chunk_overlap', 0)
        
        embed_model = HuggingFaceEmbedding(model_name=embed_model)

        transformations = [
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            embed_model
        ]

        retriever_configs = [
            BM25RetrieverConfig(),
            FAISSRetrieverConfig(dimensions=1024)
        ]

        ranker_configs = [BGERerankConfig(model=rerank_model)]

        if persist_dir.exists():
            engine = cls.from_index(
                index_config=FAISSIndexConfig(persist_path=persist_dir),
                embed_model=embed_model,
                retriever_configs=retriever_configs,
                ranker_configs=ranker_configs
            )
        else:
            engine = cls.from_docs(
                input_dir=input_dir,
                input_files=input_files,
                retriever_configs=retriever_configs,
                ranker_configs=ranker_configs,
                transformations=transformations,
                embed_model=embed_model,
            )
            engine.retriever.persist(persist_dir=persist_dir)

        return engine

    @classmethod
    def build_modular_engine(
        cls,
        input_dir: str = None, 
        input_files: List[str] = None, 
        persist_dir: Path = None,
        **kwargs
    ) -> SimpleEngine:
        """
        Builds an engine instance based on the provided configurations.

        Args:
            input_dir: Directory containing input files.
            input_files: List of input file paths.
            persist_dir: Directory to persist the index.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of SimpleEngine.
        """
        chunk_size = kwargs.pop('chunk_size', 1024)
        chunk_overlap = kwargs.pop('chunk_overlap', 20)
        
        persist_file = persist_dir / 'docstore.json'
        transformations = [SentenceSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)]
        docstore = SimpleDocumentStore()
        
        if persist_file.exists():
           modes_obj = docstore.from_persist_path(persist_file)
           nodes = list(modes_obj.docs.values())
           
        else: 
            documents = SimpleDirectoryReader(input_dir=input_dir, input_files=input_files).load_data(num_workers = 8)
            pipeline = IngestionPipeline(transformations = transformations,
                                         docstore = docstore)
            nodes = pipeline.run(documents=documents)
            pipeline.persist(persist_dir) 
        
        return cls(
            retriever=CustomRetriever(nodes),
            response_synthesizer=get_response_synthesizer(llm = RAGLLM(model_infer = LLM())),
            transformations=transformations
        )