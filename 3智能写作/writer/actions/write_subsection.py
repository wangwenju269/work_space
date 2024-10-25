from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.const import METAGPT_ROOT
from metagpt.config2 import Config

SUBSECTION_PROMPT_EN = """
# user requirements:
   To generate a comprehensive paragraph that elaborates on the given subsection heading within the provided context.
   
# the subsection heading: Begin by identifying the specific subsection heading provided.
   {heading}
   
# Contextual Preamble: Consider the introductory context that sets the stage for the subsection. 
   {contexts}
   
# Writing Guidelines:
   Step 1: Reflect on the subsection heading and how it relates to the preamble.
   Step 2: Develop a coherent paragraph that builds upon the preamble and directly addresses the subsection heading.
   Step 3: Ensure the paragraph is enriched with relevant details, examples, or explanations that enhance understanding.
   Step 4: Review the paragraph for clarity, coherence, and adherence to the subsection's focus.

# Instructions:
   1. Align with Subsection Heading:
      Ensure that your content directly corresponds to the topic outlined by the subsection heading. This alignment is crucial for maintaining coherence and relevance throughout the document.
   2. Follow Specified Headings:
      Strictly adhere to the headings provided. Each section should be crafted to specifically address and expand upon the ideas presented in these headings.
   3. Focus on Content Depth:
      Concentrate on developing the substance of your writing around the core theme indicated by the subsection heading. Where applicable, enrich your discussion with relevant references to support your arguments and enhance credibility.  
"""
       
SUBSECTION_PROMPT_ZH = """
# 用户需求
   {user_requirement}
   
# 给定的子标题：
   {heading}
   
# 任务:
   根据用户需求，生成一个详尽的段落，该段落需要详细阐述所提供的标题所涵盖的内容。
   
# 参考信息
   [文档片段]({context})
      
# 写作指南:
      步骤1: 构建一个连贯的段落，直接针对子标题展开。
      步骤2: 在主体部分，需深度挖掘子标题的内涵，通过具体事例、数据或理论分析，增强论述的说服力和可读性。
      步骤3: 审阅段落，确保清晰、连贯，并紧扣子标题的重点。
"""      
       
SUBSECTION_PROMPT =  {
                        'zh': SUBSECTION_PROMPT_ZH,
                        'en': SUBSECTION_PROMPT_EN,
                     }    
    
                        
class WriteSubsection(Action):
   # def __init__(self, context = None, llm = None):
   #       read_config = Config.read_yaml( METAGPT_ROOT / 'config/longwriter.yaml')
   #       if read_config:
   #         super().__init__(config = Config(**read_config), context = context, llm = llm)
   #       else:
   #         super().__init__(context = context, llm = llm)   
           
   async def run( self,
                  user_requirement : str , 
                  heading  : str,  
                  context : str, 
                  language : str = 'zh',
                  **kwargs) -> str:  
          
        structual_prompt = SUBSECTION_PROMPT[language].format(
                                                     user_requirement = user_requirement, 
                                                     heading = heading,
                                                     context = context
                                                    )
        prompt = self.llm.format_msg([Message(content=structual_prompt, role="user")])
        rsp = await self.llm.aask(prompt, **kwargs)
        return rsp

