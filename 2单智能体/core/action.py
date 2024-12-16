import re
from typing import Dict, Tuple
import json


class ActionParser:
    """
    Output parser for LLM response.
    """

    def __init__(self, **kwargs):
        self.tools = kwargs.get('tools', None)

    def parse_response(self, response):
        """
        Parse the LLM response to extract action information.

        Args:
            response: The LLM response to parse.

        Returns:
            Tuple[str, Dict]: A tuple containing the action name and parameters.
        """
        raise NotImplementedError

    @staticmethod
    def handle_fallback(action: str, action_para: str) -> Tuple[str, Dict]:
        """
        Handle the case where the action parsing fails.

        Args:
            action (str): The action name.
            action_para (str): The action parameters.

        Returns:
            Tuple[str, Dict]: A tuple containing the action name and parameters.

        Raises:
            ValueError: If the action is invalid.
        """
        if action and action != '':
            parameters = {'fallback': action_para}
            return action, parameters
        else:
            raise ValueError('Wrong response format for action parser')


class REACTActionParser(ActionParser):
    """
    Action parser for REACT-style LLM responses.
    """

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """
        Parse the LLM response to extract tool name and parameters.

        Args:
            response (str): The LLM response, which should conform to a predefined format.

        Returns:
            Tuple[str, Dict]: A tuple containing the tool name and parameters.
        """
        if 'Action' not in response or 'Action Input:' not in response:
            return None, None

        action, action_para = '', ''
        try:
            plugin_name, plugin_args = '', ''
            i = response.rfind('\nAction:')
            j = response.rfind('\nAction Input:')
            k = response.rfind('\nObservation:')

            # If the response has `Action` and `Action Input`, but does not contain `Observation`,
            # add it back.
            if 0 <= i < j:
                if k < j:
                    response = response.rstrip() + '\nObservation:'
                k = response.rfind('\nObservation:')

                plugin_name = response[i + len('\nAction:'):j].strip()
                plugin_args = response[j + len('\nAction Input:'):k].strip()
                response = response[:k]

            plugin_args = json.loads(plugin_args)
            return plugin_name, plugin_args
        except Exception as e:
            print(f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(action, action_para)


class OpenAiFunctionsActionParser(ActionParser):
    """
    Action parser for OpenAI function-calling LLM responses.
    """

    def parse_response(self, response: dict) -> Tuple[str, Dict]:
        """
        Parse the LLM response to extract tool name and parameters.

        Args:
            response (dict): The LLM response, which should be an OpenAI response message.
                Example:
                {
                    "content": null,
                    "function_call": {
                        "arguments": "{\n  \"location\": \"Boston, MA\"\n}",
                        "name": "get_current_weather"
                    },
                    "role": "assistant"
                }

        Returns:
            Tuple[str, Dict]: A tuple containing the tool name and parameters.
        """
        if 'function_call' not in response or response['function_call'] == {}:
            return None, None

        function_call = response['function_call']

        try:
            # Parse directly
            action = function_call['name']
            arguments = json.loads(function_call['arguments'].replace('\n', ''))
            return action, arguments
        except Exception as e:
            print(f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(function_call['name'], function_call['arguments'])