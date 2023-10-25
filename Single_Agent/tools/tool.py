TOOLS = [
        {
        'name_for_human':
            'search',
        'name_for_model':
            'search',
        'description_for_model':
            'useful for when you need to answer questions about events.',
        'parameters': [
                {
                "name": "query",
                "type": "string",
                "description": "search query of google",
                'required': True
                }
            ], 
        },

        {
        'name_for_human':
            'math',
        'name_for_model':
            'math',
        'description_for_model':
            'Useful for when you need to answer questions about Math.',
        'parameters': [
                {
                "name": "query",
                "type": "string",
                "description": "the problem to solved by math",
                'required': True
                }
            ],
        }, 
        {
        'name_for_human':
            'arxiv',
        'name_for_model':
            'arxiv',
        'description_for_model':
            "Useful when you need to answer questions about literature and scientific articles."
            "A wrapper around arxiv.org Useful for when you need to answer questions from scientific articles on arxiv.org.",
        'parameters': [
                {
                    "name": "query",
                    "type": "string",
                    "description": "the document id of arxiv to search",
                    'required': True
                }
            ],
        },

        {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            # "Note: If complex mathematical problems can be implemented using programming languages, first consider using python to solve them."
            "A Python shell.Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [
                {
                "name": "query",
                "type": "string",
                "description": "a valid python command.",
                'required': True
                }
            ], 
        },

        {   
        'name_for_human': "weather forecast",   
        'name_for_model': "weather_api",
        "description_for_model": 'weather_api is a service for obtaining weather information. The input is a city name, and real-time weather information is obtained based on the city name.',
        "parameters": [
               {
                'name': 'location',   
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
                "required": ["location"],
                }
            ]
        },

        {
            'name_for_human': 'image_gen',
            'name_for_model': 'image_gen',
            'description_for_model': 'image_gen is an AI painting (image generation) service that inputs a text description and returns the URL of the image drawn based on the text.',
            'parameters': [
                {
                    'name': 'prompt',
                    'description': 'English keywords that describe what content you want the image to have',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        }, 

        {
            'name_for_human': 'not_use_tool',
            'name_for_model': 'no_use_tool',
            'description_for_model': 'The tools are not useful for answering the question.',
            'parameters': [
                {
                    'name': 'no_use_tool',
                    'description': 'The tools are not useful for answering the question.',
                    'required': False
                }
            ],
        },
        {
        'name_for_human': 'Speech synthesis',
        'name_for_model': 'speech_synthesis',
        'description_for_model': 'Speech synthesis is a text-to-speech service that inputs a text description and returns a speech signal.',
        'parameters': [
                {
                    'name': 'prompt',
                    'description': 'Describes what exactly you want text-to-speech to contain',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ]
        }
    ]





        # {
        #     'name_for_human': '谷歌搜索',
        #     'name_for_model': 'search',
        #     'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
        #     'parameters': [
        #         {
        #             'name': 'query',
        #             'description': '搜索关键词或短语',
        #             'required': True,
        #             'schema': {'type': 'string'},
        #         }
        #     ],
        # } 
     
        # {
        #     'name_for_human': '文生图',
        #     'name_for_model': 'image_gen',
        #     'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
        #     'parameters': [
        #         {
        #             'name': 'prompt',
        #             'description': '英文关键词，描述了希望图像具有什么内容',
        #             'required': True,
        #             'schema': {'type': 'string'},
        #         }
        #     ],
        # }, 
        # {
        # 'name_for_human': '语音朗读',
        # 'name_for_model': 'text2speech',
        # 'description_for_model': '语音朗读是一个文本转化语音（语音朗读）服务，输入文本描述，朗读文本的信息',
        # 'parameters': [
        #         {
        #             'name': 'prompt',
        #             'description': '中文句子，描述了希望朗读的文本要包含具体丰富内容',
        #             'required': True,
        #             'schema': {'type': 'string'},
        #         }
        #     ]
        # }

        


