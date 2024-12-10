from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message

BEGGING_PROMPT_EN = """
# The user's requirement
  The user wants to focus on: {user_requirement}

# Chapter Overview
- **Chapter Title**: “{chapter_name}”
- **Subheadings**:
    {subheadings}
    
# Reference Information
  Please refer to the following document fragment: ({context})
   
# Task:
   Based on the user's specific requirements, integrate the chapter title, subheadings, and the provided reference content to conceptualize and write the introduction part of the chapter. The introduction should serve as a guiding role, setting a clear writing direction and tone for the development of each subheading.

# Answer Format:
  The content of the answer should be given in JSON format.

## Example Format:
{
   "Preamble":str = "Write an introduction that outlines the purpose and structure of the chapter."
}
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
   基于用户的特定需求，整合章节标题、小标题以及提供的参考内容，构思并撰写该章节的引言部分。
   引言应起到导向作用，为各小标题的展开设定明确的写作逻辑与基调,输出避免使用 **读者** 字样。
   
# 回答格式：
回答的内容请以Markdown的格式给出。

## 示例格式：
{{{chapter_name}
   <概述该章节的写作逻辑,完整连贯自然段>
}}
"""

BEGGING_PROMPT = {'zh': BEGGING_PROMPT_ZH, 'en': BEGGING_PROMPT_EN}

class WriteGuide(Action):
    async def run(self,
                  user_requirement: str,
                  chapter_name: str,
                  subheadings: str,
                  context: str,
                  language: str = 'zh'):
        
        structual_prompt = BEGGING_PROMPT[language].format(
            user_requirement=user_requirement,
            chapter_name=chapter_name,
            subheadings=subheadings,
            context=context
        )
        prompt = self.llm.format_msg([Message(content=structual_prompt, role="user")])
        content = await self.llm.aask(prompt)
        return content
   
