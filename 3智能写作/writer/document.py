import fire
from metagpt.const import DATA_PATH
from metagpt.ext.writer.roles.writer import DocumentWriter
from metagpt.ext.writer.utils.common import  WriteOutFile

REQUIREMENT = "写一个完整、连贯的《{topic}》文档, 确保文字精确、逻辑清晰，并保持专业和客观的写作风格。"
topic = "高支模施工方案"

async def main(auto_run: bool = True, use_store:bool = True):
    ref_dir     = DATA_PATH / f"ai_writer/ref/{topic}"
    persist_dir = DATA_PATH / f"ai_writer/persist/{topic}"
    output_path = DATA_PATH / f"ai_writer/outputs"
    model ='model/bge-large-zh-v1.5'
    if use_store:
       from metagpt.ext.writer.rag.retrieve import build_engine 
       store = build_engine(ref_dir,persist_dir,model)
       dw = DocumentWriter(auto_run=auto_run,store = store) 
    else:
       dw = DocumentWriter(auto_run=auto_run,use_store = False, ref_dir = ref_dir)      
    
    requirement = REQUIREMENT.format(topic = topic)
    await dw.run(requirement)
    
    # write out word
    WriteOutFile().write_markdown_file(topic = topic, 
                tasks= dw.planner.titlehierarchy.traverse_and_output(),
                output_path = output_path)
    
if __name__ == "__main__":
    fire.Fire(main)
