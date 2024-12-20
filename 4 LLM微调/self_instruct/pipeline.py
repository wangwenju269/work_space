import json
import os
import argparse
import fasttext
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import (
    MetadataMode,
    TransformComponent,
)
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llm_api import UserLLM
from answer import GetAnswer
from filter import (
    unify_format,
    url_filter,
    self_defined_rules,
    ccnet_rules,
    duplicates_rules,
)

DEFAULT_QUESTION_GEN_TMPL = '''{context_str}
请针对这篇文章，提出{num_questions}个中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...
'''

DEFAULT_TITLE_NODE_TEMPLATE = '''\
Context: {context_str}. 
根据上下文中所有独特的实体、标题或主题，生成一个简洁的标题。
Title: '''


class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = url_filter(unify_format(node.text))
        return nodes


class TextFilter(TransformComponent):
    def __call__(self, nodes, **kwargs):
        new_nodes = []
        model = fasttext.load_model(args.fasttext_model_dir)
        for node in nodes:
            self_defined_result = self_defined_rules(node.text)
            duplicates_result = duplicates_rules(node.text)
            ccnet_result = ccnet_rules(node.text, model)
            if self_defined_result[0] and duplicates_result[0] and ccnet_result[0]:
                new_nodes.append(node)
        return new_nodes


class ConstructData:
    @classmethod
    def _from_path_dir(cls, args, **kwargs):
        documents = SimpleDirectoryReader(
            input_dir=args.input_data
        ).load_data(num_workers=args.workers)
        llm = UserLLM()
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=2048, chunk_overlap=128),
                TextFilter(),
                TextCleaner(),
                TitleExtractor(
                    llm=llm,
                    node_template=DEFAULT_TITLE_NODE_TEMPLATE,
                ),
                QuestionsAnsweredExtractor(
                    llm=llm,
                    questions=5,
                    prompt_template=DEFAULT_QUESTION_GEN_TMPL,
                ),
                GetAnswer(
                    llm=llm,
                    metadata_mode=MetadataMode.NONE,
                ),
            ],
            docstore=SimpleDocumentStore(),
        )

        nodes = pipeline.run(documents=documents)

        # 相关数据写出
        data = [
            {
                "instruction": node.metadata['questions_this_excerpt_can_answer'],
                "title": node.metadata['document_title'],
                "content": node.text,
                "output": node.metadata['answer'],
            }
            for node in nodes
        ]

        # 检查输出目录是否存在，如果不存在则创建
        output_dir = os.path.exists(args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(args.output_dir)

        file_path = os.path.join(args.output_dir, 'query.json')  # 拼接文件路径
        with open(file_path, 'w') as f:
            json_str = json.dumps(data, indent=4, ensure_ascii=False)  # 将数据转换为 JSON 字符串，并格式化输出
            f.write(json_str + '\n')  # 写入文件，并在末尾添加换行符


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data construction script.")

    # 添加命令行参数
    parser.add_argument(
        "--input_data",
        help="Dataset input directory."  # 输入数据目录
    )

    parser.add_argument(
        "--fasttext_model_dir",
        help="Fasttext model directory"  # Fasttext 模型目录
    )

    parser.add_argument(
        "--output_dir",
        help="Output file directory."  # 输出文件目录
    )

    parser.add_argument(
        "--workers",
        default=8,  # 默认值为 8
        type=int,   # 参数类型为整数
        help="Multiprocessing workers' num."  # 多进程工作线程数量
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用 ConstructData 类的 _from_path_dir 方法
    ConstructData._from_path_dir(args)