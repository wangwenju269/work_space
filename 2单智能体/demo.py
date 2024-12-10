from core_agent.llm import LLMFactory
from core_agent.agent import AgentExecutor
from core_agent.agent_types import AgentType
import json

model_name = 'qwen_14b'
model_cfg_file = os.getenv('MODEL_CONFIG_FILE', 'copliot/config/cfg_model.json')
with open(model_cfg_file, 'r', encoding='utf-8') as file:
     model_cfg = json.load(file)

tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', 'copliot/config/cfg_tool.json')
with open(tool_cfg_file, 'r', encoding='utf-8') as file:
     tool_cfg = json.load(file)

llm = LLMFactory.build_llm(model_name, model_cfg)
agent = AgentExecutor(llm, tool_cfg, AgentType.REACT)
# agent.run('介绍一下你自己')
# agent.reset()
agent.run('检测<安全带.jpg>是否存在风险隐患', remote=True)








