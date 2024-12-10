import fire
from metagpt.const import DATA_PATH
from metagpt.ext.writer.roles.writer import DocumentWriter
from metagpt.ext.writer.utils.common import WriteOutFile
from metagpt.ext.writer.utils.config import AIWriterDataConfig
from metagpt.ext.writer.rag.retrieve import SimpleEngines

REQUIREMENT = "写一个完整、连贯的《数字化经济》文档, 确保文字精确、逻辑清晰，并保持专业和客观的写作风格。"
TOPIC = "数字化经济"


async def main():
    data_config = AIWriterDataConfig(rootpath=DATA_PATH, topic=TOPIC)
    if data_config.use_engine:
        engine = SimpleEngines.build_modular_engine(
            input_dir=data_config.ref_dir,
            persist_dir=data_config.persist_dir,
            **data_config.to_dict
        )
        dw = DocumentWriter(engine=engine)
    else:
        dw = DocumentWriter()
    
    requirement = REQUIREMENT.format(topic=TOPIC)
    await dw.run(requirement)
    
    WriteOutFile().write_markdown_file(
        topic=TOPIC,
        tasks=dw.planner.titlehierarchy.traverse_and_output(),
        output_path=data_config.output_path
    )


if __name__ == "__main__":
    fire.Fire(main)