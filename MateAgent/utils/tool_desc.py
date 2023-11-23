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
                "description": "Location parameter should be a single city name, e.g. San Francisco",
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
        'name_for_human': 'text_speech',
        'name_for_model': 'text_speech',
        'description_for_model': "text_speech is a text-to-speech service",
        'parameters': [
                {
                    'name': 'text',
                    'description': "'text' must be a response obtained by using the large language model to think about user questions step by step",
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ]
        }
    ]

