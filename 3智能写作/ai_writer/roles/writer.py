from __future__ import annotations
from typing import Literal,List, Optional
from pydantic import Field, model_validator
import re
from llama_index.core.schema import TextNode
from metagpt.roles import Role
from metagpt.schema import Message, Task, TaskResult
from metagpt.actions import SearchAndSummarize
from metagpt.ext.writer.roles.write_planner import WritePlanner
from metagpt.ext.writer.actions import (
     WriteGuide,
     WriteSubsection
     )
from metagpt.ext.writer.utils.common import colored_decorator, print_time



class DocumentWriter(Role):
    goal: str = "write a long document"
    auto_run: bool = True
    use_plan: bool = True
    human_design_planner: bool = True
    react_mode: Literal["plan_and_act", "react"] = 'plan_and_act'
    max_react_loop: int = 3
    planner: WritePlanner = Field(default_factory=WritePlanner)
    engine: Optional[object] = Field(default=None, exclude=True)


    @model_validator(mode="after")
    def set_plan(self):
        self._set_react_mode(
            react_mode=self.react_mode,
            max_react_loop=self.max_react_loop,
            auto_run=self.auto_run
        )
        
        self.planner = WritePlanner(
            auto_run=self.auto_run,
            human_design_planner=self.human_design_planner
        )
        
        self.use_plan = (self.react_mode == "plan_and_act")
        self.set_actions([WriteGuide, WriteSubsection])
        self._set_state(0)
        
        return self


    @model_validator(mode="after")
    def validate_store(self):
        self.search = SearchAndSummarize(context=self.context)
        return self


    @property
    def working_memory(self):
        return self.rc.working_memory
    
    @print_time
    async def run(self, requirement) -> Message | None:
        return await super().run(requirement)
        
        
    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """
        Process a given task by either writing an initial draft or refining a draft based on user feedback.
        This method differentiates the action to be taken based on the completion status of the task.
        If the task is marked as finished, it will refine the draft; otherwise, it will write a new draft.
        """
        chapter_id = re.sub(r'\.\d+', '', current_task.task_id)  # 章节序号
        
        await self.recursive_write_draft(chapter_id)  # 写本段落初稿
        
        current_task.is_finished = True
        task_result = TaskResult(result='', is_success=True)
        
        return task_result
    
    
    async def recursive_write_draft(
        self,
        chapter_id: str
    ) -> None:
        """
        This function is responsible for creating an initial draft of a chapter based on provided information.
        """
        chapter_name = self.planner.titlehierarchy.get_chapter_obj_by_id(chapter_id).name  # 父标题
        subheadings = self.planner.titlehierarchy.get_subheadings_by_prefix(chapter_id)  # 子标题
        instruction = self.planner.titlehierarchy.get_chapter_obj_by_id(chapter_id).instruction or chapter_name   # 当前写作指令
        self.working_memory.add(
            Message(
                content=instruction,
                role="user",
                cause_by=f'{chapter_id} {chapter_name}'
            )
        )

        if subheadings:
            # Inner node
            await self.recursive_generate_paragraph(subheadings)
            self._set_state(0)
            content = await self.generate_guide(
                chapter_id,
                chapter_name,
                subheadings,
                instruction=instruction,
                context='\n'.join([str(self.working_memory.index[x]) for x in subheadings])
            )
        else:
            # Leaf node
            self._set_state(1)
            content = await self.generate_paragraph(
                chapter_id,
                chapter_name,
                instruction=instruction
            )

        self.planner.titlehierarchy.set_content_by_id(chapter_id, content)
        self.working_memory.add(
            Message(
                content=content,
                role="assistant",
                cause_by=f'{chapter_id} {chapter_name}'
            )
        )

        await self.add_nodes_document(chapter_name,content)

    
    async def recursive_generate_paragraph(self, headings: List[str]):
        for chapter in headings:
            match = re.match(r'^\d+(\.\d+)*', chapter)
            if match:
               chapter_id = match.group(0)
            else:
                try:
                        chapter_id, _ = chapter.split(' ', 1)
                except ValueError as e:
                    print(f"ValueError: Unable to split chapter '{chapter}' into chapter_id and rest. Error: {e}")
                        
            await self.recursive_write_draft(chapter_id)
    
    
    async def generate_guide(
        self,
        chapter_id: str,
        chapter_name: str,
        subheadings: List[str],
        instruction: str = '',
        context: str = ''
    ) -> str:
        """
        Generate the beginning guidelines of a chapter based on the current task and user requirements.
        """
        subheadings_str = ','.join([section for section in subheadings])
        
        content = await self.rc.todo.run(
            user_requirement=instruction,
            chapter_name=f'{chapter_id} {chapter_name}',
            subheadings=subheadings_str,
            context=context
        )
        
        return content
    
    
    async def generate_paragraph(
        self,
        chapter_id: str,
        chapter_name: str,
        instruction: str = ''
    ) -> str:
        
        information = await self.node_retrieve(instruction)
        
        content = await self.rc.todo.run(
            user_requirement=instruction,
            heading=f'{chapter_id} {chapter_name}',
            context=information
        )
        
        return content
        

    @colored_decorator("\033[1;46m")
    async def node_retrieve(self,query:str) -> str:
        """
        Retrieve relevant information from documents based on the given query.
        """
        if self.engine:
            response = await self.engine.aquery(query)     
            return response.response
        else:
            return ''


    async def add_nodes_document(self,chapter_name,content):  
        if hasattr(self.engine, 'retriever') and hasattr(self.engine.retriever, 'add_nodes'):
           node = TextNode(text = content, metadata = {'title': chapter_name})
           self.engine.retriever.add_nodes([node])
              