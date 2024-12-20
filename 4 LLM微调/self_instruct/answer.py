from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.llms.llm import LLM
from llama_index.core.settings import Settings
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)

SUBSECTION_PROMPT_ZH = """
# 用户需求
   {instruction}
# 给定标题：
   {title}
# 任务:
   根据用户需求，生成一个详尽的段落，该段落需要详细阐述所提供的标题所涵盖的内容。
# 参考信息
   [文档片段]({content}) 
# 写作指南:
      步骤1: 构建一个连贯的段落，直接针对标题展开。
      步骤2: 在主体部分，需深度挖掘子标题的内涵，通过具体事例、数据或理论分析，增强论述的说服力和可读性。
      步骤3: 审阅段落，确保清晰、连贯，并紧扣子标题的重点。
"""


class GetAnswer(BaseExtractor):
    """
    Questions answered extractor. Node-level extractor.
    Extracts `questions_this_excerpt_can_answer` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        prompt_template (str): template for response answer
    """

    llm: LLMPredictorType = Field(description="The LLM to use for generation.")
    prompt_template: str = Field(
        default=SUBSECTION_PROMPT_ZH,
        description="Prompt template to use when generating questions.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLMPredictorType] = None,
        prompt_template: str = SUBSECTION_PROMPT_ZH,
        embedding_only: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            prompt_template=prompt_template,
            embedding_only=embedding_only,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GetAnswer"

    async def _aget_answer_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract answers from a node and return its metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}
        context_str = node.get_content(metadata_mode=self.metadata_mode)
        prompt = PromptTemplate(template=self.prompt_template)
        answer = await self.llm.apredict(
            prompt,
            instruction=node.metadata['document_title'],
            title=node.metadata['questions_this_excerpt_can_answer'],
            content=context_str,
        )
        return {"answer": answer.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        questions_jobs = []
        for node in nodes:
            questions_jobs.append(self._aget_answer_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            questions_jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        return metadata_list