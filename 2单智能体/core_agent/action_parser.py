import re
from typing import Dict, Tuple

import json
from core_agent.agent_types import AgentType


class ActionParser:
    """Output parser for llm response
    """

    def __init__(self, **kwargs):
        self.tools = kwargs.get('tools', None)

    def parse_response(self, response):
        raise NotImplementedError

    # use to handle the case of false parsing the action_para result, if there is no valid action then
    # throw Error
    @staticmethod
    def handle_fallback(action: str, action_para: str):
        if action is not None and action != '':
            parameters = {'fallback': action_para}
            return action, parameters
        else:
            raise ValueError('Wrong response format for action parser')


class REACTActionParser(ActionParser):
    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'Action' not in response or 'Action Input:' not in response:
            return None, None
        action, action_para = '', ''
        try:
            plugin_name, plugin_args = '', ''
            i = response.rfind('\nAction:')
            j = response.rfind('\nAction Input:')
            k = response.rfind('\nObservation:')
            if 0 <= i < j:  # If the response has `Action` and `Action input`,
                if k < j:  # but does not contain `Observation`,
                    response = response.rstrip() + '\nObservation:'  # Add it back.
                k = response.rfind('\nObservation:')
                plugin_name = response[i + len('\nAction:') : j].strip()
                plugin_args = response[j + len('\nAction Input:') : k].strip()
                response = response[:k]
            plugin_args = json.loads(plugin_args)    
            return plugin_name, plugin_args
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(action, action_para)






class OpenAiFunctionsActionParser(ActionParser):
    def parse_response(self, response: dict) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters
        Args:
            response (str): llm response, it should be an openai response message
            such as
            {
                "content": null,
                "function_call": {
                  "arguments": "{\n  \"location\": \"Boston, MA\"\n}",
                  "name": "get_current_weather"
                },
                "role": "assistant"
            }
        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """
        if 'function_call' not in response or response['function_call'] == {}:
            return None, None
        function_call = response['function_call']

        try:
            # parse directly
            action = function_call['name']
            arguments = json.loads(function_call['arguments'].replace(
                '\n', ''))

            return action, arguments
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(function_call['name'],
                                                function_call['arguments'])


