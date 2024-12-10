import os 
from typing import List
from metagpt.actions import Action
from metagpt.schema import Message
from collections import defaultdict
from llama_index.core.schema import TextNode
from metagpt.ext.writer.utils.common import colored_decorator

PROMPT_TEMPLATE_ZH = """注意：你的回答必须严格遵循知识库内容，即使与事实不符。
如果知识库的大部分内容都与问题无关，只有少数几句话与问题直接相关，请重点关注这几句话，这种情况一定要回复。

# 知识库

{ref_doc}"""

PROMPT_TEMPLATE_EN = """Please respond solely based on the content of the provided Knowledge Base.
Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
If the materials mainly contains content irrelevant to the question, with only a few sentences directly related, please focus on these sentences and ensure a response.

# Knowledge Base

{ref_doc}"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}

PROMPT_END_TEMPLATE_ZH = """# 问题
{question}


# 回答规则
1. 信息密度与篇幅匹配：
  - 确保回答内容的信息密度高，全面覆盖知识库中的相关知识点，输出篇幅应与知识库中信息量相匹配。
2. 严格依据知识库内容：
  - 回答时必须严格遵守知识库内容，即使这可能与外界事实有所出入。回答的准确性以知识库为准。
3. 结构化回答：
  - 采用“分要点回答”模式，确保每个关键点都有充足的解释和信息支持，避免遗漏重要细节。
4. 聚焦相关性：
  - 当知识库中大部分内容与问题关联不大时，应特别关注并详细回答与问题直接相关的部分，确保这些关键信息得到充分阐述。
  
请根据回答规则，针对知识库内容回答问题，请详细回答："""

PROMPT_END_TEMPLATE_EN = """# Question
{question}


# Answering Guidelines
- Please respond solely based on the content of the provided Knowledge Base.
- Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
- If the materials mainly contains content irrelevant to the question, with only a few sentences directly related, please focus on these sentences and ensure a response.

Please give your answer:"""

PROMPT_END_TEMPLATE = {
    'zh': PROMPT_END_TEMPLATE_ZH,
    'en': PROMPT_END_TEMPLATE_EN,
}

KNOWLEDGE_TEMPLATE_ZH = """# 知识库

{knowledge}"""

KNOWLEDGE_TEMPLATE_EN = """# Knowledge Base

{knowledge}"""

KNOWLEDGE_TEMPLATE = {'zh': KNOWLEDGE_TEMPLATE_ZH, 'en': KNOWLEDGE_TEMPLATE_EN}

KNOWLEDGE_SNIPPET_ZH = """## 来自 {source} 的内容：

```
{content}
```"""

KNOWLEDGE_SNIPPET_EN = """## The content from {source}:

```
{content}
```"""

KNOWLEDGE_SNIPPET = {'zh': KNOWLEDGE_SNIPPET_ZH, 'en': KNOWLEDGE_SNIPPET_EN}

class GenSummary(Action):
    
    @colored_decorator("\033[1;46m")
    async def run(self,
                  title: str, 
                  nodes: List[TextNode],
                  lang: str = 'zh', 
                  **kwargs
                  ) -> str:
            knowledge = self.process_knowledge(nodes)
            system_prompt = PROMPT_TEMPLATE[lang].format(ref_doc=knowledge)
            prompt = PROMPT_END_TEMPLATE[lang].format(question = title)
            context = self.llm.format_msg([Message(content=prompt, role="user")])
            rsp = await self.llm.aask(context, system_msgs = [system_prompt], **kwargs)
            return rsp
        
    @staticmethod
    def process_knowledge(nodes: List[TextNode],lang = 'zh') -> str:
        knowledge =  defaultdict(list)
        for id,  node in enumerate(nodes):
            content = f'[page: {id + 1}]\n\n{node.text}'
            file_name = os.path.basename(node.metadata['file_path'])
            source = f'[文件]({file_name})'   
            knowledge[f'{source}'].append(content)
        knowledges = []
        for source, snippet in knowledge.items(): 
            snippets = KNOWLEDGE_SNIPPET[lang].format(source = source, content = '\n\n...\n\n'.join(snippet))
            knowledges.append(snippets)
        return '\n\n'.join(knowledges) 