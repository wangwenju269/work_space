from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.const import METAGPT_ROOT
from metagpt.config2 import Config

SUBSECTION_PROMPT_EN = """
# User Requirement
   {user_requirement}
   
# Given Subheading:
   {heading}
   
# Task:
   Generate a detailed paragraph that thoroughly elaborates on the provided subheading based on the user's requirement.
   
# Reference Information
   [Document Fragment]({context})
      
# Writing Guidelines:
   Step 1: Construct a coherent paragraph that directly addresses the subheading.
   Step 2: In the main body, delve deeply into the subheading's implications, using specific examples, data, or theoretical analysis to enhance the persuasiveness and readability of the argument.
   Step 3: Review the paragraph to ensure clarity, coherence, and focus on the subheading's key points.
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
    def __init__(self, context):
        super().__init__(context = context)
        self.set_config(self._load_config())

    def _load_config(self):
        config_path = METAGPT_ROOT / 'config/longwriter.yaml'
        if config_path.exists():
            return Config.from_home(config_path)
        return None

    async def run(self, user_requirement: str, heading: str, context: str, language: str = 'zh', **kwargs) -> str:
        structural_prompt = SUBSECTION_PROMPT[language].format(
            user_requirement=user_requirement,
            heading=heading,
            context=context
        )
        prompt = self.llm.format_msg([Message(content=structural_prompt, role="user")])
        rsp = await self.llm.aask(prompt, **kwargs)
        return rsp

