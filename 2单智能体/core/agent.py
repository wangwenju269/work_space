import importlib
from typing import Dict, List, Optional, Union

from .action import ActionParser, REACTActionParser
from .prompt.prompt import PromptGenerator
from .retrieve import (SUPPORTED_KNOWLEDGE_TYPE, KnowledgeRetrieval,
                       ToolRetrieval)


class AgentExecutor:
    """
    The core class of the agent. It is responsible for the interaction between the user, LLM, and tools,
    and returns the execution result to the user.
    """

    def __init__(self,
                 llm,
                 tool_cfg: Optional[Dict] = {},
                 agent_type: str = 'react',
                 additional_tool_list: Optional[Dict] = {},
                 prompt_generator: Union[PromptGenerator, str] = None,
                 action_parser: Union[ActionParser, str] = None,
                 tool_retrieval: Optional[Union[bool, ToolRetrieval]] = False,
                 knowledge_retrieval: Optional[KnowledgeRetrieval] = None,
                 **kwargs):
        """
        Initialize the AgentExecutor.

        Args:
            llm: The LLM model, which can be loaded from a local or remote server.
            tool_cfg (Optional[Dict]): Configuration for default tools.
            agent_type (str): The type of agent. Defaults to 'react'.
            additional_tool_list (Optional[Dict]): User-defined additional tools. Defaults to {}.
            prompt_generator (Optional[PromptGenerator]): The module responsible for generating prompts
                based on interaction results. Defaults to using PromptGenerator.
            action_parser (Optional[ActionParser]): The module responsible for parsing LLM output
                into executable actions. Defaults to using REACTActionParser.
            tool_retrieval (Optional[Union[bool, ToolRetrieval]]): Retrieve related tools based on the input task.
                If it is a boolean and True, the default ToolRetrieval will be used. Defaults to False.
            knowledge_retrieval (Optional[KnowledgeRetrieval]): If the user wants to use extra knowledge,
                this component can be used to retrieve related knowledge. Defaults to None.
        """
        self.llm = llm
        self.agent_type = agent_type
        self._init_tools(tool_cfg, additional_tool_list)

        if isinstance(tool_retrieval, bool) and tool_retrieval:
            tool_retrieval = ToolRetrieval()
        self.tool_retrieval = tool_retrieval
        if self.tool_retrieval:
            self.tool_retrieval.construct(
                [str(t) for t in self.tool_list.values()])

        self.knowledge_retrieval = knowledge_retrieval
        self.prompt_generator = prompt_generator or PromptGenerator()
        self.action_parser = action_parser or REACTActionParser()
        self.reset()
        self.seed = None

    def _init_tools(self, tool_cfg: Dict = {}, additional_tool_list: Dict = {}):
        """
        Initialize the tool list for the agent. A default tool list is provided, which is initialized
        from a configuration file. Users can also provide custom tools via additional_tool_list.

        Args:
            tool_cfg (Dict): Configuration for default tools.
            additional_tool_list (Dict): User-defined tools. Defaults to {}.
        """
        self.tool_list = {}
        tool_info_list = {**additional_tool_list}
        tools_module = importlib.import_module('core_agent.tools')

        for tool_name in tool_cfg.keys():
            if tool_cfg[tool_name].get('use', False):
                assert tool_name in tool_info_list, f'Invalid tool name: {tool_name}, ' \
                    f'available ones are: {tool_info_list.keys()}'
                tool_class_name = tool_info_list[tool_name]
                tool_class = getattr(tools_module, tool_class_name)
                tool_name = tool_class.name
                self.tool_list[tool_name] = tool_class(tool_cfg)

        self.tool_list = {**self.tool_list, **additional_tool_list}
        self.set_available_tools(self.tool_list.keys())

    def set_available_tools(self, available_tool_list):
        """
        Set the available tools for the agent.

        Args:
            available_tool_list (List[str]): List of tool names.
        """
        for t in available_tool_list:
            if t not in self.tool_list:
                raise ValueError(
                    f'Unsupported tools found: {t}, valid ones: {self.tool_list.keys()}')

        self.available_tool_list = {
            k: self.tool_list[k]
            for k in available_tool_list
        }

    def retrieve_tools(self, query: str) -> List[str]:
        """
        Retrieve tools based on the query.

        Args:
            query (str): The query to retrieve tools.

        Returns:
            List[str]: List of available tools.
        """
        if self.tool_retrieval:
            retrieve_tools = self.tool_retrieval.retrieve(query)
            self.set_available_tools(available_tool_list=retrieve_tools.keys())
        return self.available_tool_list.values()

    def get_knowledge(self, query: str, append_files: list = []) -> List[str]:
        """
        Retrieve knowledge based on the query.

        Args:
            query (str): The query to retrieve knowledge.
            append_files (list): User-provided files to append during runtime.

        Returns:
            List[str]: List of retrieved knowledge.
        """
        append_files = [
            item for item in append_files
            if item.endswith(tuple(SUPPORTED_KNOWLEDGE_TYPE))
        ]

        if append_files:
            if not self.knowledge_retrieval:
                self.knowledge_retrieval = KnowledgeRetrieval.from_file(append_files)
            else:
                self.knowledge_retrieval.add_file(append_files)

        return self.knowledge_retrieval.retrieve(query) if self.knowledge_retrieval else []

    def run(self,
            task: str,
            remote: bool = False,
            print_info: bool = False,
            append_files: list = [],
            **kwargs) -> List[Dict]:
        """
        Use the LLM and tools to execute the task provided by the user.

        Args:
            task (str): The concrete task.
            remote (bool): Whether to execute tools in remote mode. Defaults to False.
            print_info (bool): Whether to print prompt information. Defaults to False.
            append_files (list): The list of files to append to knowledge or refer to.

        Returns:
            List[Dict]: The execution result. A task may require multiple interactions with the LLM,
            so a list of dictionaries is returned. Each dictionary contains the result of one interaction.
        """
        # Retrieve tools and knowledge
        tool_list = self.retrieve_tools(task)
        knowledge_list = self.get_knowledge(task, append_files)

        self.prompt_generator.init_prompt(
            task, tool_list, knowledge_list, append_files=append_files)
        function_list = self.prompt_generator.get_function_list(tool_list)

        llm_result, exec_result = '', ''
        idx = 0
        final_res = []

        while True:
            idx += 1

            # Generate prompt and call LLM
            llm_artifacts = self.prompt_generator.generate(llm_result, exec_result)
            try:
                llm_result = self.llm.generate(llm_artifacts, function_list)
            except RuntimeError as e:
                return [{'exec_result': str(e)}]

            # Parse and get tool name and arguments
            try:
                action, action_args = self.action_parser.parse_response(llm_result)
            except ValueError as e:
                return [{'exec_result': f'{e}'}]

            if action is None:
                # In chat mode, update the final result to the prompt history
                _ = self.prompt_generator.generate(llm_result, '')
                return final_res

            if action in self.available_tool_list:
                action_args = self.parse_action_args(action_args)
                tool = self.tool_list[action]

                # Handle special case for image generation
                if action == 'image_gen' and self.seed:
                    action_args['seed'] = self.seed

                try:
                    exec_result = tool(**action_args, remote=remote)
                    final_res.append(exec_result)
                    self.parse_exec_result(exec_result)
                except Exception as e:
                    exec_result = f'Action call error: {action}: {action_args}. \n Error message: {e}'
                    return [{'exec_result': exec_result}]
            else:
                exec_result = f"Unknown action: '{action}'. "
                return [{'exec_result': exec_result}]

    def reset(self):
        """
        Clear the history and agent state.
        """
        self.prompt_generator.reset()
        self.agent_state = {}

    def parse_action_args(self, action_args):
        """
        Replace action arguments in string format with Image/Video/Audio wrappers,
        so that tools can handle them.

        Args:
            action_args (Dict): The action arguments.

        Returns:
            Dict: Parsed action arguments.
        """
        parsed_action_args = {}
        for name, arg in action_args.items():
            try:
                true_arg = self.agent_state.get(arg, arg)
            except Exception as e:
                true_arg = arg
            parsed_action_args[name] = true_arg
        return parsed_action_args

    def parse_exec_result(self, exec_result, *args, **kwargs):
        """
        Update the execution result to the agent state.

        Args:
            exec_result (Dict): The execution result.
        """
        for k, v in exec_result.items():
            self.agent_state[str(v)] = v