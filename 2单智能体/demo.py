import os
import json
from core.agent import AgentExecutor
from core.llm.openai import OpenAi
from core.llm.model import ModelLLM

# 获取模型配置文件路径
model_cfg_file = os.getenv('MODEL_CONFIG_FILE', './config/cfg_model.json')
with open(model_cfg_file, 'r', encoding='utf-8') as file:
    model_cfg = json.load(file)

# 获取工具配置文件路径
tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', './config/cfg_tool.json')
with open(tool_cfg_file, 'r', encoding='utf-8') as file:
    tool_cfg = json.load(file)


def build_llm(model_name, cfg):
    """
    根据模型名称和配置构建 LLM 实例。

    Args:
        model_name (str): 模型名称。
        cfg (dict): 模型配置。

    Returns:
        LLM: 构建的 LLM 实例。
    """
    llm_type = cfg[model_name].pop('type')
    llm_cfg = cfg[model_name]

    if llm_type == 'openai':
        llm = OpenAi(cfg=llm_cfg)
    else:
        llm = ModelLLM(cfg=llm_cfg)

    return llm


# 构建 LLM 实例
llm = build_llm('qwen', model_cfg)

# 初始化 AgentExecutor
agent = AgentExecutor(llm, tool_cfg, 'react')

# 运行任务
agent.run('介绍一下你自己')