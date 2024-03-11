from core_agent.agent_types import AgentType
from .react_prompt import ReactPromptGenerator

class PromptGeneratorFactory:

    @classmethod
    def get_prompt_generator(cls,
                             agent_type: AgentType = AgentType.DEFAULT,
                             model = None,
                             **kwargs):
        
        if AgentType.REACT == agent_type:
            return ReactPromptGenerator(**kwargs)
        else:
            raise NotImplementedError

