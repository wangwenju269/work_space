from typing import List, Tuple,Union
import json ,re 
from llama_index.core.schema import TextNode
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT
NO_RESPONSE = '<None>'
SYSTEM_PROMPT_TEMPLATE_ZH = """您是一个擅长文档问答的专家，可以根据文档内容回答用户问题。

# 任务描述：
请仔细阅读所给的文档片段，并根据其内容回答问题。
您需要判断文档的内容是否可以回答问题，不要强行回答。
如果可以回答，答案必须严格遵循文档内容，即使与事实不符。
如果答案与事实不符，直接返回空，不要做解释。


# 回答规则：
- 评估文档是否含有足够信息回答问题。无关时一定不要回答。
- 如果问题能被回答，你的回答必须严格遵循文档内容，即使与事实不符。一定不要做多余解释。
- 如果问题能被回答，直接引用文档的相关信息保证答案准确、完整，并追求简洁。
- 当文档中只有少量信息与问题相关时，重点关注这部分信息，这种情况下一定回答。

# 回答格式：
回答的内容请以JSON的格式给出。


## 示例：
当文档内容无关时：
{{"res": "none", "content": "[]", "relevance_level": "无"}}
Observation: ...

当文档内容可回答，且文档为中文时：
{{"res": "ans", "content": "[你的答案]","relevance_level": "高 | 中 | 低"}}
Observation: ...

当文档内容可回答，且文档为英文时：
{{"res": "ans", "content": "[Your Answer]","relevance_level": "High | Medium | Low"}}
"""

SYSTEM_PROMPT_TEMPLATE_EN = """You are an expert in document-based question answering, capable of answering user questions based on document content.

# Task Description:
Please read the provided document excerpt carefully and answer questions based on its content.
You must assess whether the document content allows for the questions to be answered, without forcing a response.
If the answer does not align with the facts, provide it directly without explanation.


# Answering Rules:
- Reply in the same language as the source material.
- Evaluate whether the document contains sufficient information to answer the question. Do not respond if it's irrelevant.
- If the question can be answered, your answer must strictly follow the document content, even if it does not align with the facts.
- If the question can be answered, directly quote the relevant information from the document to ensure the answer is accurate, complete, and strive for conciseness.
- When the document contains only minimal information related to the question, focus on this information and be sure to answer.


# Answer Format:
Please provide answers in the form of JSON.


## Examples
When the document content is irrelevant:
{{"res": "none", "content": "{no_response}"}},
Observation: ...

When the document content can provide an answer:
{{"res": "ans", "content": "[Your Answer]"}}
Observation: ..."""

SYSTEM_PROMPT_TEMPLATE = {
    'zh': SYSTEM_PROMPT_TEMPLATE_ZH,
    'en': SYSTEM_PROMPT_TEMPLATE_EN,
}

PROMPT_TEMPLATE_ZH = """# 文档：
{ref_doc}

# 问题：
{instruction}

请根据回答规则，给出你的回答："""

PROMPT_TEMPLATE_EN = """# Document:
{ref_doc}

# Question:
{instruction}

Please provide your answer according to the answering rules:"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}

DEFAULT_LENS = 1024

class RelateFilter(Action):
    
    def __init__(self):
        read_config = Config.read_yaml( METAGPT_ROOT / 'config/qwen2-7B.yaml')
        super().__init__(config = Config(**read_config) if read_config else None )
       
           
    async def run(  self,
                    instruction: str = None,
                    knowledge: str = '',
                    lang: str = 'zh',
                    **kwargs) -> str :
        
        system_prompt = SYSTEM_PROMPT_TEMPLATE[lang].format(no_response=NO_RESPONSE)
        prompt = PROMPT_TEMPLATE[lang].format(ref_doc=knowledge, instruction=instruction)
        context = self.llm.format_msg([Message(content=prompt, role="user")])
        rsp = await self.llm.aask(context, system_msgs = [system_prompt], stream = False, **kwargs)
        return rsp
    
  
    def parser_out_filter_nodes(self,
                                nodes : Union[TextNode,str] , 
                                rsp : str ,
                                batch_size : int = 5
                                )-> Tuple[List[TextNode], List[str]]: 
        new_nodes, pre_answer = [], [] 
        for ans, node in zip(rsp, nodes):
            try:
                content_dict = json.loads(ans.replace("\n", "").replace('\r', ''))
                pa_res, pa_cotent, relevance_level = content_dict['res'], content_dict['content'] , content_dict['relevance_level']
                if  relevance_level == '高'  or  relevance_level == 'High':
                    new_nodes.append(node) 
                    pre_answer.append(pa_cotent)
                if pa_res == 'none':
                    continue   
            except:
                pa_cotent = re.sub(r'[{}"]|("res":\s*"ans"|"res":\s*"none"|"\s*content":\s*)', '', ans)
                flag = 'res: ans, content:'
                if pa_cotent.startswith(flag):
                   new_nodes.append(node) 
                   pre_answer.append(pa_cotent.replace(flag,'').strip())
                
        if  not new_nodes :  
            return nodes,  ''
        combined = ['\ncontext:'.join(pre_answer[i:i + batch_size]) for i in range(0, len(pre_answer), batch_size)]  
        return  new_nodes , combined
    

