from __future__ import annotations
import json
from pydantic import BaseModel, Field
from metagpt.actions.di.ask_review import AskReview, ReviewConst
from metagpt.strategy.planner import Planner
from metagpt.actions.di.write_plan import (
    precheck_update_plan_from_rsp,
    update_plan_from_rsp,
)
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import Message, Plan, Task, TaskResult
from metagpt.utils.common import remove_comments
from metagpt.actions import Action
from metagpt.utils.common import CodeParser
from user_develop.prompt.report_task_type import UserTaskType

STRUCTURAL_CONTEXT = """
## User Requirement
{user_requirement}
## Context
{context}
## Current Plan
{tasks}
## Current Task
{current_task}
"""

PLAN_STATUS = """
您的目标是构建一个完整、连贯的段落，为整个报告奠定基调。请确保您的文本内容准确、有逻辑性。
请记住，您的输出将直接影响报告的初步印象，因此请保持专业和客观的风格。

当前任务：{current_task}

已生成的章节段落：{report_written}。

为了更好地协助您完成任务，以下是一些指导性建议:
{guidance}

"""

class WritePlan(Action):
    PROMPT_TEMPLATE: str = """
    # Context:
    {context}
    # Available Task Types:
    {task_type_desc}
    # Task:
    Arrange the plan strictly in paragraph order.
    Cannot arrange the same paragraph repeatedly
    Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan consists of one to {max_tasks} tasks.
    If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole plan unless instructed to modify only one task of the plan.
    If you encounter errors on the current task, revise and output the current single task only.
   
    Output a list of jsons following the format:
     ```json
    [
        {{
            "task_id": str = "unique identifier for a task in plan, can be an ordinal",
            "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
            "instruction": "what you should do in this task, one short phrase or sentence",
            "task_type": "type of this task, should be one of Available Task Types"
        }},
        ...
    ]
    ```
    """
    async def run(self, context: list[Message], max_tasks: int = 7, human_design_sop = True) -> str:
        if not human_design_sop:
            task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in UserTaskType])
            prompt = self.PROMPT_TEMPLATE.format(
                context="\n".join([str(ct) for ct in context]), max_tasks=max_tasks, task_type_desc=task_type_desc
            )
            rsp = await self._aask(prompt)
            rsp = CodeParser.parse_code(block=None, text=rsp)
        else:
            from metagpt.const import DATA_PATH              
            with    open(DATA_PATH / 'sop.json', 'r', encoding='utf-8') as file:
                    rsp = json.dumps(json.loads(file.read()), ensure_ascii=False)    # 读取人类设计 `sop` 流程 
        return rsp



class WritePlanner(Planner):
    human_design_sop: bool = False
    # 不改变原有代码结构的基础上, 重写这段方法代码
    async def update_plan(self, goal: str = "", max_tasks: int = 7, max_retries: int = 3):
        if goal:
            self.plan = Plan(goal=goal)
        plan_confirmed = False
        while not plan_confirmed:
            context = self.get_useful_memories()
            rsp = await WritePlan().run(context, max_tasks, self.human_design_sop)
            self.working_memory.add(Message(content=rsp, role="assistant", cause_by=WritePlan))
            # precheck plan before asking reviews
            is_plan_valid, error = precheck_update_plan_from_rsp(rsp, self.plan)
            if not is_plan_valid and max_retries > 0:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.warning(error_msg)
                self.working_memory.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                max_retries -= 1
                continue
            _, plan_confirmed = await self.ask_review(trigger=ReviewConst.TASK_REVIEW_TRIGGER)
        update_plan_from_rsp(rsp=rsp, current_plan=self.plan)
        # working_memory 
        self.working_memory.clear()


    def get_plan_status(self) -> str:
        # prepare components of a plan status
        finished_tasks = self.plan.get_finished_tasks()
        paragraph_written = [remove_comments(task.code) for task in finished_tasks]
        paragraph_written = "\n\n".join(paragraph_written)
        task_results = [task.result for task in finished_tasks]
        task_results = "\n\n".join(task_results)
        task_type_name = self.current_task.task_type.lower()
        guidance = UserTaskType[task_type_name].value.guidance if hasattr(UserTaskType, task_type_name) else ""

        # combine components in a prompt
        prompt = PLAN_STATUS.format(
            report_written = paragraph_written,
            validate_result = task_results,
            current_task=self.current_task.instruction,
            guidance=guidance,
        )
        return prompt
