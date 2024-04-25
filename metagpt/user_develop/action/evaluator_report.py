from __future__ import annotations

import json
from metagpt.actions import Action
from metagpt.schema import Message, Plan
from metagpt.utils.common import CodeParser, process_message, remove_comments
STRUCTUAL_PROMPT = """请根据 `User Requirement` ,`Plan Status`, `Tool Info`,`Reference Information`字段信息对 `report`文本信息进行评估。
评估点如下：
    1.语言质量：评估生成的文本是否通顺、语法正确、表达清晰，以及是否符合预期的主题和风格。
    2.内容准确性：评估生成的文本是否包含准确、全面的信息，以及是否能够准确地回答问题或完成任务。
    3.逻辑性：评估生成的文本是否具有合理的逻辑关系等。
    4.上下文相关性：评估生成的文本是否与给定的上下文信息相关，以及是否能够根据上下文信息进行恰当的推理和推断。
    5.安全性：评估生成的文本是否包含敏感信息或不适宜的内容，以及是否能够避免产生歧视性、攻击性或恶意的言论。

# report 
{report}

# User Requirement
{user_requirement}

# Plan Status
{plan_status}

# Tool Info
{tool_info}

# Reference Information
{info}

 Output a list of jsons following the format:
     ```json
    [
        {{
            "Evaluation_point": str = "按照 5 个评价要点的先后顺序,依次评价 `report` 里文本信息",
            "critique": str = "`report` 生成文本优化方向",
            "score":int =  "评估分数,分数区间在 0-5 之间"
        }},
        ...
    ]
    ```
"""

class EvaluatorReport(Action):
    async def run(
        self,
        report:str,
        user_requirement: str,
        plan_status: str = "",
        tool_info: str = "",
        working_memory: list[Message] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        while max_retries:
            try :
                info = "\n".join([str(ct) for ct in working_memory if str(ct).startswith('user')]) if working_memory else ''
                structual_prompt = STRUCTUAL_PROMPT.format(
                    report= report,
                    user_requirement=user_requirement,
                    plan_status=plan_status,
                    tool_info= tool_info,
                    info = info
                )
                context = process_message([Message(content=structual_prompt, role="user")])
                rsp_report = await self.llm.aask(context, **kwargs)
                rsp_report = CodeParser.parse_code(block=None, text = rsp_report) 
                rsp  =  json.loads(rsp_report)
                all_score  = sum([x['score'] for x in rsp]) / len(rsp) 
                return rsp_report, True if all_score >= 4.5  else False
            except:   
                max_retries -= 1  
        return  rsp_report, False
        
        