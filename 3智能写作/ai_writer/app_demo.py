import gradio as gr
import json
import os
import shutil
from typing import Optional
from pydantic import Field

from metagpt.const import DATA_PATH
from metagpt.schema import Message, Plan
from metagpt.actions import SearchAndSummarize, UserRequirement
from metagpt.ext.writer.roles.write_planner import DocumentPlan, WritePlanner
from metagpt.ext.writer.actions import WriteGuide, Refine, WriteSubsection
from metagpt.ext.writer.rag.retrieve import SimpleEngines
from metagpt.ext.writer.utils.common import WriteOutFile

class DocumentGenerator(WritePlanner):
    store: Optional[object] = Field(default=None, exclude=True)

    def add_file_button(self, topic, history, add_file_button):
        ref_dir = DATA_PATH / f"ai_writer/ref/{topic}"
        persist_dir = DATA_PATH / f"ai_writer/persist/{topic}"

        if not os.path.isdir(ref_dir):
            os.makedirs(ref_dir)

        for file in add_file_button:
            shutil.copy(file, ref_dir)
            history.append(('finish upload', file))

        self.store = SimpleEngines.build_modular_engine(
            input_dir=ref_dir,
            persist_dir=persist_dir
        )

        shutil.rmtree(ref_dir)
        return history

    async def generate_outline(self):
        context = self.get_useful_memories()
        response = await DocumentPlan().run(context)
        return response

    async def gen_outline_button(self, requirements: str):
        self.plan = Plan(goal=requirements.strip())
        response = await self.generate_outline()
        return [(requirements, response)]

    async def submit_outline_button(self, user_input, conversation_history):
        self.working_memory.add(Message(content=user_input, role="user"))
        response = await self.generate_outline()
        conversation_history.append((user_input, response))
        return "Outline updated", conversation_history

    def confirm_outline_button(self, requirements, history, outline):
        self.plan = Plan(goal=requirements.strip())
        if not outline:
            outline = history[-1][-1] if history else ''
        '''根据大纲建文档目录树状结构'''
        rsp = self.post_process_chapter_id_or_name(outline)
        self.process_response_and_build_hierarchy(rsp=rsp)
        return outline

    def get_name_and_subheading(self, id):
        obj = self.titlehierarchy.get_chapter_obj_by_id(id)
        chapter_name = obj.name
        subheadings = self.titlehierarchy.get_subheadings_by_prefix(id)
        return chapter_name, '\n'.join(subheadings)

    async def retrieve_button(self, chapter_name):
        contexts = ''
        if self.store:
            contexts = await self.store.aretrieve(chapter_name)
            contexts = '\n\n'.join([x.text for x in contexts])
        return contexts

    async def gen_guide(self, chapter_id, chapter_name, subheadings, history):
        if subheadings:
            context = await self.retrieve_button(chapter_name)
            result = await WriteGuide().run(
                user_requirement=self.plan.goal,
                chapter_name=chapter_name,
                subheadings=','.join([section for section in subheadings]),
                context=context
            )
            history.append((f'{chapter_id} {chapter_name}', result))
            yield history

            for subheading in subheadings:
                chapter_id, name = subheading.split(' ')
                subtitle = self.titlehierarchy.get_subheadings_by_prefix(chapter_id)
                async for output in self.gen_guide(chapter_id, name, subtitle, history):
                    yield output

    async def gen_guide_button(self, chapter_id, history):
        history = []
        subheadings = self.titlehierarchy.get_subheadings_by_prefix(chapter_id)
        chapter_name = self.titlehierarchy.get_chapter_obj_by_id(chapter_id).name
        async for output in self.gen_guide(chapter_id, chapter_name, subheadings, history):
            yield output

    async def write_paragraph(self, parent_id, child_id, chapter_name, subheadings, history):
        if subheadings:
            content = self.titlehierarchy.get_chapter_obj_by_id(parent_id).content
            history.append((f'{child_id} {chapter_name}', content))
            yield history

            for subheading in subheadings:
                child_id, chapter_name = subheading.split(' ')
                child_heading = self.titlehierarchy.get_subheadings_by_prefix(child_id)
                async for output in self.write_paragraph(parent_id, child_id, chapter_name, child_heading, history):
                    yield output
        else:
            information = await self.retrieve_button(chapter_name)
            instruction = self.titlehierarchy.get_chapter_obj_by_id(parent_id).instruction
            if not instruction:
                instruction = self.get_useful_memories()
            gen_paragraph = await WriteSubsection().run(
                user_requirement=instruction,
                heading=f'{child_id} {chapter_name}',
                context=information
            )
            history.append((f'{child_id} {chapter_name}', gen_paragraph))
            yield history

    async def write_paragraph_button(self, chapter_id, history):
        history = []
        subheadings = self.titlehierarchy.get_subheadings_by_prefix(chapter_id)
        chapter_name = self.titlehierarchy.get_chapter_obj_by_id(chapter_id).name
        async for output in self.write_paragraph(chapter_id, chapter_id, chapter_name, subheadings, history):
            yield output

    async def web_button(self, revise_id, instrution):
        chapter_name = self.titlehierarchy.get_chapter_obj_by_id(revise_id.lstrip()).name
        prompt = instrution if instrution else chapter_name
        message = [Message(content=prompt, role="user", cause_by=UserRequirement)]
        addition_context = await SearchAndSummarize().run(message)
        return addition_context

    def download_button(self, topic):
        output_path = DATA_PATH / f"ai_writer/outputs"
        WriteOutFile().write_word_file(
            topic=topic,
            tasks=self.titlehierarchy.traverse_and_output(),
            output_path=output_path
        )

        return gr.DownloadButton(label=f"Download", value=output_path / f'{topic}.docx', visible=True)

    def create_directory_structure_botton(self, outline):
        outline = json.loads(outline)

        def generate_tree(chapter, level=0):
            indent = "........" * level
            yield f"{indent}| {chapter['chapter_id']} {chapter['chapter_name']}"
            for subheading in chapter['subheadings']:
                yield from generate_tree(subheading, level + 1)

        tree_output = []
        for chapter in outline:
            tree_output.extend(generate_tree(chapter))
        return [('目录结构', '\n'.join(tree_output))]


async def main(doc_gen = DocumentGenerator()):
    with gr.Blocks(css="") as demo:
        gr.Markdown("## AI 智能文档写作 Demo")
        with gr.Row():
            with gr.Column(scale=0, elem_id="row1"):
                with gr.Tab("开始"):
                    gr.Markdown("developed by wangwenju")
                    topic = gr.Textbox(
                        "数字化经济",
                        label="话题",
                        lines=7,
                        interactive=True,
                    )
                    user_requriments = gr.Textbox(
                        "写一个完整、连贯的《数字化经济》文档, 确保文字精确、逻辑清晰，并保持专业和客观的写作风格，中文书写",
                        label="用户需求",
                        lines=9,
                        interactive=True,
                    )
                    add_file_button = gr.UploadButton("📁 Upload (上传文件)", file_count="multiple")
                    gen_outline_button = gr.Button("生成大纲")

                with gr.Tab("大纲"):
                    outline_box = gr.Textbox(
                        label="大纲",
                        lines=16,
                        interactive=True
                    )
                    user_input = gr.Textbox(
                        'eg:请帮我新增章节',
                        lines=2,
                        label='大纲修订(增删改)'
                    )
                    submit_outline_button = gr.Button("修订")
                    confirm_outline_button = gr.Button("确认")

                with gr.Tab("生成段落"):
                    chapter_id = gr.Textbox(
                        '1',
                        label="chapter_id",
                        lines=1,
                        interactive=True
                    )
                    chapter_name = gr.Textbox(
                        '',
                        label="大章节名称",
                        lines=1,
                        interactive=False
                    )
                    chapter_subname = gr.Textbox(
                        '',
                        label="小节名称",
                        lines=2,
                        interactive=False
                    )
                    retrieve_bot = gr.Textbox(
                        '',
                        label="资源检索",
                        lines=5,
                        interactive=False
                    )
                    retrieve_button = gr.Button("资源检索")
                    gen_guide_button = gr.Button("生成指南")
                    write_paragraph_button = gr.Button("生成段落")

                with gr.Tab("功能区"):
                    instrution = gr.Textbox(
                        label="润色指令",
                        lines=4,
                        interactive=True
                    )
                    addition_context = gr.Textbox(
                        label="临时新增内容",
                        lines=10,
                        interactive=True
                    )
                    web_button = gr.Button("联网")
                    download_button = gr.DownloadButton(
                        "下载",
                        visible=True,
                    )

            with gr.Column(scale=3, elem_id="row2"):
                chatbot = gr.Chatbot(label='output', height=690)

            with gr.Column(scale=0, elem_id="row3"):
                revise_text = gr.Textbox(
                    label="修订",
                    lines=30,
                    interactive=True,
                    show_copy_button=True
                )

            add_file_button.upload(
                doc_gen.add_file_button,
                inputs=[topic, chatbot, add_file_button],
                outputs=[chatbot],
                show_progress=True
            )

            gen_outline_button.click(
                doc_gen.gen_outline_button,
                inputs=[user_requriments],
                outputs=[chatbot],
                show_progress=True
            )

            submit_outline_button.click(
                doc_gen.submit_outline_button,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot]
            )

            confirm_outline_button.click(
                doc_gen.confirm_outline_button,
                inputs=[user_requriments, chatbot, outline_box],
                outputs=[outline_box]
            ).then(
                doc_gen.create_directory_structure_botton,
                inputs=[outline_box],
                outputs=[chatbot]
            )

            retrieve_button.click(
                doc_gen.get_name_and_subheading,
                inputs=[chapter_id],
                outputs=[chapter_name, chapter_subname]
            ).then(
                doc_gen.retrieve_button,
                inputs=[chapter_name],
                outputs=[retrieve_bot]
            )

            gen_guide_button.click(
                doc_gen.get_name_and_subheading,
                inputs=[chapter_id],
                outputs=[chapter_name, chapter_subname]
            ).then(
                doc_gen.gen_guide_button,
                inputs=[chapter_id, chatbot],
                outputs=[chatbot]
            )

            write_paragraph_button.click(
                doc_gen.write_paragraph_button,
                inputs=[chapter_id, chatbot],
                outputs=[chatbot]
            )

            web_button.click(
                doc_gen.web_button,
                inputs=[chapter_id, instrution],
                outputs=[addition_context],
                show_progress=True
            )

            download_button.click(
                doc_gen.download_button,
                inputs=[topic],
                outputs=download_button,
                show_progress=True
            )

    demo.queue().launch(
        share=True,
        inbrowser=False,
        server_port=8771,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
