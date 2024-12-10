import json
from typing import List
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.ext.writer.utils.common import colored_decorator

PROMPT_TEMPLATE_ZH = """请提取问题中的关键词，需要中英文均有，可以适量补充不在问题中但相关的关键词。关键词尽量切分为动词、名词、或形容词等单独的词，不要长词组（目的是更好的匹配检索到语义相关但表述不同的相关资料）。关键词以JSON的格式给出，比如{{"keywords_zh": ["关键词1", "关键词2"], "keywords_en": ["keyword 1", "keyword 2"]}}

Question: 这篇文章的作者是谁？
Keywords: {{"keywords_zh": ["作者"], "keywords_en": ["author"]}}
Observation: ...

Question: 解释下图一
Keywords: {{"keywords_zh": ["图一", "图 1"], "keywords_en": ["Figure 1"]}}
Observation: ...

Question: 核心公式
Keywords: {{"keywords_zh": ["核心公式", "公式"], "keywords_en": ["core formula", "formula", "equation"]}}
Observation: ...

Question: {user_request}
Keywords:
"""

PROMPT_TEMPLATE_EN = """Please extract keywords from the question, both in Chinese and English, and supplement them appropriately with relevant keywords that are not in the question.
Try to divide keywords into verb, noun, or adjective types and avoid long phrases (The aim is to better match and retrieve semantically related but differently phrased relevant information).
Keywords are provided in JSON format, such as {{"keywords_zh": ["关键词1", "关键词2"], "keywords_en": ["keyword 1", "keyword 2"]}}

Question: Who are the authors of this article?
Keywords: {{"keywords_zh": ["作者"], "keywords_en": ["author"]}}
Observation: ...

Question: Explain Figure 1
Keywords: {{"keywords_zh": ["图一", "图 1"], "keywords_en": ["Figure 1"]}}
Observation: ...

Question: core formula
Keywords: {{"keywords_zh": ["核心公式", "公式"], "keywords_en": ["core formula", "formula", "equation"]}}
Observation: ...

Question: {user_request}
Keywords:
"""


PROMPT_TEMPLATE = {
        'zh': PROMPT_TEMPLATE_ZH,
        'en': PROMPT_TEMPLATE_EN,
    }


class GenKeyword(Action):
    
    @colored_decorator("\033[1;46m")
    async def run(self, 
                  messages: str,
                  lang: str = 'zh', 
                  **kwargs
                  ) -> str :
            prompt = PROMPT_TEMPLATE[lang].format(user_request=messages)
            context = self.llm.format_msg([Message(content=prompt, role="user")])
            rsp = await self.llm.aask(context, **kwargs)
            rsp = self.parser_out(rsp, messages, lang)
            return rsp
    
    def parser_out(self, 
                   keyword : str, 
                   messages : str,
                   lang : str = 'zh',
                   ) -> List[str]:
        
        keyword = keyword.strip()
        if keyword.startswith('```json'):
            keyword = keyword[len('```json'):]
        if keyword.endswith('```'):
            keyword = keyword[:-3]
        try:
            keyword_dict = json.loads(keyword)
            key_words = keyword_dict['keywords_zh'] if lang == 'zh' else keyword_dict['keywords_en']
            key_words = ', '.join(key_words)
            return key_words
        except Exception:
            return messages