import json5
class   GET_TOOL_DESC:
        TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""
        @classmethod
        def get_tools_text(cls,list_of_plugin_info) -> str:
            tools_text = []
            for plugin_info in list_of_plugin_info:
                tool = cls.TOOL_DESC.format(
                    name_for_model=plugin_info["name_for_model"],
                    name_for_human=plugin_info["name_for_human"],
                    description_for_model=plugin_info["description_for_model"],
                    parameters=json5.dumps(plugin_info["parameters"], ensure_ascii=False),
                )
                if plugin_info.get('args_format', 'json') == 'json':
                    tool += " Format the arguments as a JSON object."
                elif plugin_info['args_format'] == 'code':
                    tool += ' Enclose the code within triple backticks (`) at the beginning and end of the code.'
                else:
                    raise NotImplementedError
                tools_text.append(tool)
            tools_text = '\n\n'.join(tools_text)
            tools_name_text = ', '.join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])
            return tools_text,  tools_name_text
         
        # 换一种写法
        # tool_desc = """{name_for_model}:What is the {name_for_model} API useful for? {description_for_model} Parameters: {parameters}"""
        @classmethod
        def get_tools_other_text(cls,list_of_plugin_info) -> str:
            tools_text = {}
            for plugin_info in list_of_plugin_info:
                tool = cls.TOOL_DESC.format(
                    name_for_model=plugin_info["name_for_model"],
                    name_for_human=plugin_info["name_for_human"],
                    description_for_model=plugin_info["description_for_model"],
                    parameters= plugin_info["parameters"]
                )
                i = tool.index(':')
                tool_name , tool_desc = tool[:i],tool[i+1:]
                tools_text.update({tool_name:tool_desc}) 
            tools_text = json5.dumps(tools_text,ensure_ascii= False)
            tools_name_text = [plugin_info["name_for_model"] for plugin_info in list_of_plugin_info]
            return tools_text, tools_name_text




