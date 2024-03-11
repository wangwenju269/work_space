from core_agent.agent_types import AgentType
from core_agent.llm import LLM
from .action_parser import REACTActionParser
                            
class ActionParserFactory:

    @classmethod
    def get_action_parser(cls,
                          agent_type: AgentType = AgentType.DEFAULT,
                          model: LLM = None,
                          **kwargs):
        if AgentType.REACT == agent_type:
            return REACTActionParser(**kwargs)
        else:
            raise NotImplementedError



