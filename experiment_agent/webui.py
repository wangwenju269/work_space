import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from argparse import ArgumentParser
from pathlib import Path
import copy
import gradio as gr
import shutil
import re
import secrets
import tempfile
import warnings
warnings.filterwarnings("ignore")
from config.parser import DataArguments
from llm.model import Qwen  
from tools.plugins import User_defined_tools
from agents import React

PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＝＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
strat_momey = []

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=True,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")
    args = parser.parse_args()
    return args

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def launch_demo(arg):
    # uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    uploaded_file_dir = '/data/wangwj-t/workspace/experiment/data'
    args = DataArguments()
    qwen = Qwen(args.checkpoint)
    External_API =  User_defined_tools(llm = qwen,
                                      args = args
                                       ) 

    react_agent =   React(  External_API = External_API,
                            llm = qwen
                             )

    """ 记得清空环境里文件信息"""
    file_path_folder = args.file_folder
    def del_env_path(file_path_folder):
        ls = os.listdir(file_path_folder)
        for s in ls :
            f_path = os.path.join(file_path_folder, s)
            if os.path.isdir(f_path):
               del_env_path(f_path) 
            else:   
               os.remove(f_path)
    del_env_path(file_path_folder)


    def upload_env_path(file_path,args):
        file_names = os.path.basename(file_path)
        _ , suffix = file_names.split('.')
        if  suffix in ['pdf']:
            shutil.move(file_path, args.pdf_file)
        elif suffix  in ['png','jpg']:
            shutil.move(file_path, args.image_file)
        else:
            raise  ValueError("{0} format is not supported, Only supports[/.pdf/.jpg/.png]formats ".format(suffix))




    def predict(_chatbot, task_history):
        print(f"task_history:\n{task_history}")
        chat_query = _chatbot[-1][0]
        query =      task_history[-1][0]
        print("User: " + parse_text(query))
        history_cp = copy.deepcopy(task_history)
        history_cp = history_cp[-2:] if len(history_cp) > 2 else history_cp    # 只保留两轮对话
        full_response = ""
        history_filter = []
        pre = ""
        for q, a in history_cp:
            if isinstance(q, (tuple, list)):
                file_path = q[0]
                upload_env_path(file_path,args)  
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        print(f'history:\n{history}')
        chat_response = react_agent.run(message, history=history)  # 'chat_stream'
        chat_response = react_agent.reason_path

        "推理完成结束，记住把 推理链 清空"
        react_agent.reset() 
        # if  filename :
        #     chat_response = chat_response.replace("<ref>", "").replace(r"</ref>", "")
        #     chat_response = re.sub(BOX_TAG_PATTERN, "" , chat_response)
        #     if chat_response != "":
        #         _chatbot[-1] = (parse_text(chat_query), chat_response)
        #     _chatbot.append((None, (str(filename),)))

        # else:
        _chatbot[-1] = (parse_text(chat_query), chat_response)
        full_response = parse_text(chat_response)
        task_history[-1] = (query, full_response)
        # print(f"chatbot:\n{_chatbot}")
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        '环境清空'
        External_API.reset()
        task_history.clear()
        return []

    
    def post(task_history):
        return strat_momey


    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>Agent</center>""")
        gr.Markdown(
            """\
<center><font size=3>This Agent is based on Qwen, developed by Wang wenju. \
(本Agent实现聊天机器人调用外部工具的功能。)</center>""")
        gr.Markdown("""\
<center><font size=3>工具可选集合: search,math,safety_belt_checker,python,weather_api,image_gen,speech_synthesis \
</center>""")

        chatbot = gr.Chatbot(label='Agent', elem_classes="control-height", height=550)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State(strat_momey)
        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True)

        submit_btn.click(reset_user_input, [], [query])
        # empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True).then(post,[task_history], [task_history],show_progress=False)

        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=arg.share,
        inbrowser=arg.inbrowser,
        server_port=arg.server_port,
        server_name=arg.server_name
    )

if __name__ == '__main__':
    launch_demo(_get_args())

