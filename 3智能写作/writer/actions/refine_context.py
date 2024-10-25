from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message

REFINE_EN = """
# user's guidance:
   {user_requirement}
   
# original answer:
   {respones}   
   
# Additional Context: 
   {contexts}    
   
# task:
   Enhance the clarity and effectiveness of the original answer to the query “{original_query}” by incorporating the user's guidance and the additional context provided.

# Constraints:
1. If the additional context proves irrelevant or lacks substance, default to providing the initial response without modification.
2. Ensure that the refined answer is coherently integrated and maintains a high level of logical flow and readability.

# Instructions:
- Carefully review the user's guidance to understand the desired improvements.
- Assess the additional context for its relevance and potential to enhance the original answer.
- Modify the original answer by adding, removing, or rephrasing content as necessary to align with the user's guidance and the new context.
- Proofread the refined answer to ensure it is clear, concise, and effectively addresses the original query.
- Maintain the integrity of the original response while incorporating the enhancements.
"""


REFINE_ZH = """
# 用户指令:
   {user_requirement}
   
# 原始答案:
   {respones}   
   
# 附加信息: 
   {contexts}    
   
# 任务:
   请根据用户的指导和提供的附加信息，改进对原始问题 “{original_query}” 的回答，以提高其清晰度和有效性。

# 约束条件:
   1.如果附加信息不相关或缺乏实质内容，则默认使用初始回答不做修改。
   2.根据需要通过添加、删除或重新措辞内容来修改原始答案，使修改后答案符合用户指令
"""

REFINE = {
    'zh': REFINE_ZH,
    'en': REFINE_EN,
}

class Refine(Action):
    async def run(  self,
                    original_query: str,
                    respones : str,
                    contexts  : str,
                    user_requirement : str = '',
                    language : str = 'zh',
                    **kwargs) -> str:    
        
        original_query = f"{original_query}"  
        structual_prompt = REFINE[language].format(
            original_query = original_query,
            respones = respones,
            contexts = contexts,
            user_requirement = user_requirement
            )
        context = self.llm.format_msg([Message(content=structual_prompt, role="user")])
        rsp = await self.llm.aask(context, **kwargs)
        return rsp