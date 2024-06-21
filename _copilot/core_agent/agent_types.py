from enum import Enum


class AgentType(str, Enum):

    DEFAULT = 'default'
    """"""
    REACT = 'react'
    """An agent that does a reasoning step before acting with react"""

