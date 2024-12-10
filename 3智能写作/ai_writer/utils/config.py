from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

@dataclass
class AIWriterDataConfig:
      rootpath: Path = field(
         default=Path("."),
         metadata={"help": "The root directory where all data and outputs are stored."},
      )
      topic: Optional[str] = field(
         default="default",
         metadata={"help": "The topic for AI_writer."},
      )
      use_engine: bool = field(
         default=True,
         metadata={"help": "Whether to use the retrieval engine."},
      )
      auto_run: bool = field(
         default= True,
         metadata={"help": "Whether to use the auto run code."},
      )
      embed_model: Optional[str] = field(
         default="model/BAAI/bge-large-zh-v1.5",
         metadata={"help": "The path to the embedding model to be used."},
      )
      rerank_model: Optional[str] = field(
         default="model/BAAI/bge-reranker-large",
         metadata={"help": "The path to the reranking model to be used."},
      )

      @property
      def ref_dir(self) -> Path:
         return self.rootpath / "ai_writer/ref" / self.topic

      @property
      def persist_dir(self) -> Path:
         return self.rootpath / "ai_writer/persist" / self.topic

      @property
      def output_path(self) -> Path:
         return self.rootpath / "ai_writer/outputs"
      
      @property
      def to_dict(self) -> dict:
        return asdict(self)