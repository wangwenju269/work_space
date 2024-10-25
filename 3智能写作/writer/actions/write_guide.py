from __future__ import annotations
import json
from metagpt.actions import Action
from metagpt.schema import Message
BEGGING_PROMPT_EN = """
# user requirements
   {user_requirement}
# the current chapter title
   {chapter_name}
# the subheadings
   {subheadings}
# Selective reference
   {contexts}
# Task:
   Based on the user's specified writing topic and considering the current chapter title, its subheadings, and selective reference, 
   assist in crafting an introductory preamble for this chapter. 
   The preamble should serve as a guiding statement that sets the stage for the content to follow, highlighting the central theme and providing a roadmap for the subheadings.

# Constraint:
   1. Integrate insights from the subheadings to enrich the content.
   2. Keep the expansion to approximately 10 sentences for brevity.
   3. Ensure the response is clear and concise, avoiding unnecessary elaboration.
"""

BEGGING_PROMPT_ZH = """
# 用户需求
  用户希望聚焦于：{user_requirement}

# 章节概览
- **章节标题**：“{chapter_name}”
- **章节小标题**:
    {subheadings}
   
# 参考信息
  请参考以下文档片段：({context})
   
# 任务:
   基于用户的特定需求，整合章节标题、小标题以及提供的参考内容，构思并撰写该章节的引言部分。引言应起到导向作用，为读者铺设理解路径，并为各小标题的展开设定明确的写作方向与基调。

# 回答格式：
回答的内容请以JSON的格式给出。

## 示例格式：
{{
   "Preamble":str = "撰写引言，概述章节目的与结构。",
   "Guideline":LIst[Dict[str,str]] = [
         {{
            "Subtitle":"小标题一",
            "Instructions":"描述如何围绕此小标题进行深入探讨，包括关键点、示例或理论基础。"
         }},
         // 更多小标题及其指导原则...
      ] 
}}   
"""

BEGGING_PROMPT = {
    'zh': BEGGING_PROMPT_ZH,
    'en': BEGGING_PROMPT_EN,
}

class WriteGuide(Action):
   async def run(  self,
                    user_requirement: str,
                    chapter_name : str,
                    subheadings  : str,
                    context : str,
                    language : str = 'zh'
                    ) -> str:   
           
        structual_prompt = BEGGING_PROMPT[language].format(
            user_requirement=user_requirement,
            chapter_name = chapter_name,
            subheadings =  subheadings,
            context = context
            )
        
        prompt = self.llm.format_msg([Message(content=structual_prompt, role="user")])
        rsp = await self.llm.aask(prompt)
        rsp = self.parser_out(rsp)
        return rsp
   
   def parser_out(self, rsp : str):
       try:
         rsp = json.loads(rsp)
       except:
         rsp = None 
       return rsp      