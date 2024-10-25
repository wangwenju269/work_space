from __future__ import annotations
from typing import Literal,List, Optional, Dict
import re
from pathlib import Path
from pydantic import Field, model_validator
from llama_index.core.schema import TextNode
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.tools.search_engine import SearchEngine
from metagpt.schema import Message, Task, TaskResult
from metagpt.actions import SearchAndSummarize, UserRequirement 
from metagpt.ext.writer.utils.common import colored_decorator, print_time
from metagpt.ext.writer.roles.write_planner import WritePlanner
from metagpt.ext.writer.actions import (
     WriteGuide,
     WriteSubsection,
     GenSummary)
from metagpt.ext.writer.rag.retrieve import key_word_retrieve
from metagpt.ext.writer.rag.qwen_rag import qwen_request

class DocumentWriter(Role):
    name: str = "wangs"
    profile: str = "document_writer"
    goal: str = "write a long document"
    auto_run: bool = True
    use_plan: bool = True
    react_mode: Literal["plan_and_act", "react"] =  'plan_and_act'    #"by_order"
    human_design_planner: bool = True
    max_react_loop : int = 3
    planner: WritePlanner = Field(default_factory= WritePlanner)
    use_store: bool = False
    ref_dir : Path =  None
    store: Optional[object] = Field(default=None, exclude=True) 
    store_mode: Literal["search", "gen_key_word", "qwen_rag" ] = "search"
    use_write_guide : bool = True
    
    @model_validator(mode="after")
    def set_plan(self):
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop, auto_run=self.auto_run)
        self.planner = WritePlanner(auto_run=self.auto_run,
                                    human_design_planner = self.human_design_planner
                                    )
        
        self.use_plan = (self.react_mode == "plan_and_act" )      
        self.set_actions([WriteGuide, WriteSubsection])
        self._set_state(0)
        return self
    
    @model_validator(mode="after")
    def validate_stroe(self):
        if self.store:
            search_engine = SearchEngine.from_search_func(search_func=self.store.asearch, proxy=self.config.proxy)
            action = SearchAndSummarize(search_engine=search_engine, context=self.context)
        else:
            action = SearchAndSummarize(context=self.context)
        self.actions.append(action)    
        return self
    
    @property
    def working_memory(self):
        return self.rc.working_memory
    
    @print_time
    async def run(self, requirement) -> Message | None:
        return await super().run(requirement)
        
    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """
        Process a given task by either writing an initial draft or refining a draft based on user feedback .
        This method differentiates the action to be taken based on the completion status of the task.
        If the task is marked as finished, it will refine the draft; otherwise, it will write a new draft.
        """
        chapter_id = re.sub(r'\.\d+', '', current_task.task_id)     # 章节序号    
        chapter_name = self.planner.titlehierarchy.get_chapter_obj_by_id(chapter_id).name      # 父标题         
        subheadings = self.planner.titlehierarchy.get_subheadings_by_prefix(chapter_id)        # 子标题          
        await self.recursive_write_draft(chapter_id, chapter_name, subheadings)      # 写本段落初稿 
        current_task.is_finished = True      
        task_result = TaskResult(result = '', is_success = True)
        return task_result
    
    
    async def recursive_write_draft(self,
                                        chapter_id:str,
                                        chapter_name:str = '',
                                        subheadings:List[str] = [] 
                                    ) -> None:
        """
        this function is responsible for creating an initial draft of a chapter based on provided information.
        """
        self._set_state(0) 
        if  subheadings and self.use_write_guide: 
            await self.generate_guide(chapter_id, chapter_name, subheadings)
        #  a leaf node   
        if  not subheadings:
            subheadings = f"{chapter_id} {chapter_name}"    
        self._set_state(1) 
        await self.recursive_generate_paragraph(subheadings)
        # Add the generated document to the node for easier retrieval
        await self.add_nodes_doc(chapter_id)
        logger.info(f"All actions for chapter {chapter_name} have been finished.")
    
    async def recursive_generate_paragraph(self,
                                            headings : List[str] | str ,
                                          ):  
   
        if  not isinstance(headings, list):
            headings =  [headings]   
        for chapter in headings:
            chapter_id, chapter_name = chapter.split(' ')
            subheadings = self.planner.titlehierarchy.get_subheadings_by_prefix(chapter_id)
            if  subheadings :
                await self.recursive_write_draft(chapter_id, chapter_name, subheadings)
            else:    
                await self.generate_paragraph(chapter_id, chapter_name, subheadings)
        return  
    
    async def generate_guide(self,
                                chapter_id:str, 
                                chapter_name:str,
                                subheadings:List[str],
                                information:str = ''
                             ) :  
        """
        Generate the beginning guidelines of a chapter based on the current task and user requirements.
        """
        if  not hasattr(self, 'rc') or not hasattr(self.rc, 'todo'):
            raise AttributeError("Expected 'rc' and 'rc.todo' to be initialized") 
        todo = self.rc.todo  
        instruction = self.planner.titlehierarchy.get_chapter_obj_by_id(chapter_id).instruction
        if not instruction:
           instruction = self.get_memories()[0].content
        subheadings =  ','.join([section for section in subheadings])
        # information = await self.doc_retrieve(f'{chapter_name}_{subheadings}')
        result = await todo.run(
                                    user_requirement= instruction,
                                    chapter_name = chapter_name,
                                    subheadings = subheadings,
                                    context = information
                                )
        if result:
            if 'Preamble'  in result and result['Preamble']:
                self.planner.titlehierarchy.set_content_by_id(chapter_id, result['Preamble'])
            if  not self.human_design_planner and 'Guideline' in result and result['Guideline']: 
                try:
                    for item in result['Guideline']:
                        subtitle, instrution = item['Subtitle'], item['Instructions']
                        _id,  _ = subtitle.split(' ')
                        self.planner.titlehierarchy.set_instruction_by_id(_id, instrution)
                except:  pass        
        return  
    

    async def generate_paragraph(self,
                                   chapter_id:str, 
                                   chapter_name:str,
                                   subheadings:List[str] = [],
                                   information : str = '',
                                ) -> Dict[str, str]: 
            
        if  not hasattr(self, 'rc') or not hasattr(self.rc, 'todo'):
            raise AttributeError("Expected 'rc' and 'rc.todo' to be initialized") 
        todo = self.rc.todo  
        logger.info(f"ready to {todo.name}")
        instruction = self.planner.titlehierarchy.get_chapter_obj_by_id(chapter_id).instruction
        if not instruction:  
           instruction = self.get_memories()[0].content
        information  = await self.doc_retrieve(f'{instruction}_{chapter_name}')
        result = await todo.run(
                                        user_requirement = instruction, 
                                        heading = f'{chapter_id} {chapter_name}',
                                        context =  information
                                        )
        self.planner.titlehierarchy.set_content_by_id(chapter_id, result)
        return  
    

    @colored_decorator("\033[1;46m")
    async def doc_retrieve(self,title:str) -> str:
        """
        Retrieve relevant information from documents based on the given title.

        This method retrieves contexts from documents using an engine, which could be a retrieval or a RAG (Retrieval Augmented Generation) model.
        The retrieval mode is determined by the `rag_mode` attribute of the class instance. 
        The retrieved contexts are then optionally cleaned if the mode is set to 'retrieve_gen'.
        Parameters:
        title (str): The title or query used to retrieve relevant information from documents.
        Returns:
        contexts: A string containing the retrieved contexts, separated by section dividers.
        """
        if  not self.store:
            if self.store_mode == 'qwen_rag':  
               return qwen_request( text = title,  file_folder = self.ref_dir)
            else:
                return ''
        if self.store_mode == 'search':
            todo = self.actions[-1]
            message = [Message(content= title, role="user", cause_by = UserRequirement)]
            return  await todo.run(self.rc.history  + message)     

        elif  self.store_mode == 'gen_key_word':
            if   hasattr(self.store, 'retriever') and hasattr(self.store.retriever, '_nodes'):
                 pass
            elif hasattr(self.store, 'retriever') and hasattr(self.store.retriever, 'retrievers'):  
                 for retriever in self.store.retriever.retrievers: 
                     if 'bm25' in  retriever.__repr__() and hasattr(retriever, '_nodes'):
                          self.store.retriever = retriever    # 关键词检索只适合 BM25 检索
                          break
            else:    
                raise AttributeError("gen_key_word is suitable for use in the BM25 retrieval strategy") 
            nodes =  await key_word_retrieve(self.store, title)
            context = await GenSummary().run(title, nodes = nodes)
            return context
        else:
            return ''


    async def add_nodes_doc(self,chapter_id) -> None:  
        """
        Add nodes to the document retriever engine based on the assistant's messages from memory.
        This method extracts text chunks from the working memory where the role is "assistant",
        creates Document objects with these chunks, and then adds these documents as nodes to the
        retriever engine.
        """
        if  not self.use_store or not self.store:
            return 
        text_chunks = self.planner.titlehierarchy.get_child_content_by_prefix(chapter_id) 
        nodes = []
        for text in text_chunks:
            doc = TextNode(text=text, metadata = {'file_path': 'gen_node'})
            nodes.append(doc)
        # Add the list of Document objects as nodes to the retriever engine
        self.store.retriever.add_nodes(nodes)
        return        