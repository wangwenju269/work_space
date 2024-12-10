from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message

REFINE_EN = """
# User Instruction:
   {instruction}
   
# Initial Answer:
   {context}   
   
# Part to Refine:
   {select}  
   
# Task:
   Refine the specified part of the initial answer while keeping the overall content unchanged, to enhance the clarity and effectiveness of this section.

# Constraints:
   1. Only refine the specified part (i.e., the part to refine) of the initial answer; do not alter any other parts.
   2. The refined answer must meet the requirements of the user instruction.
"""

REFINE_ZH = """
# 用户指令：
   {instruction}
   
# 初始答案:
   {context}   
   
# 初始答案中待润色部分：
   {select}  
   
# 任务:
   在保持初始答案整体不变的情况下，仅对初始答案中的待润色部分进行润色操作，从而提升这部分内容的清晰度与有效性。

# 约束条件:
   1.仅针对初始答案中的指定部分（即待润色部分）进行修订，其余部分不得改动。
   2.润色后的答案需满足用户指令要求。
"""

CONTINUE_EN = """
# User Instruction:
   {instruction}
# Initial Answer:
   {context}   

# Task:
   Please perform a **continuation** of the initial answer while maintaining its overall structure and content.

# Constraints:
   1. Only the initial answer should be continued; no other parts should be modified.
   2. The continued content must meet the requirements of the user instruction.
"""

CONTINUE_ZH = """
# 用户指令：
   {instruction}
# 初始答案:
   {context}   

# 任务:
   请在保持初始答案整体结构和内容不变的前提下，对初始答案进行**续写**操作。

# 约束条件:
   1. 仅对初始答案进行续写，其他部分不得修改。
   2. 续写后的内容需符合用户指令的要求。
"""

REFINE = {'zh': REFINE_ZH, 'en': REFINE_EN}
CONTINUE = {'zh': CONTINUE_ZH, 'en': CONTINUE_EN}


class Refine(Action):
    async def polish(self,
                     instruction: str,
                     context: str,
                     select: str,
                     language: str = 'zh') -> str:
        structural_prompt = REFINE[language].format(
            instruction=instruction,
            context=context,
            select=select,
            language=language
        )
        context = self.llm.format_msg([Message(content=structural_prompt, role="user")])
        rsp = await self.llm.aask(context)
        return rsp

    async def continue_write(self,
                             instruction: str,
                             context: str = '',
                             select: str = '',
                             language: str = 'zh',
                             **kwargs) -> str:
        structural_prompt = CONTINUE[language].format(
            instruction=instruction,
            context=context,
            select=select,
            language=language
        )
        context = self.llm.format_msg([Message(content=structural_prompt, role="user")])
        rsp = await self.llm.aask(context, **kwargs)
        return rsp