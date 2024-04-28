# Prompt for taking on "eda" tasks
FIRST_PROMPT = """
请根据以下关键信息撰写一个合理的事故报告开头的第一自然段,主要介绍事故概要。在段落的结尾处，加入一段官方声明，声明应体现事故调查的严谨性和全面性。
*** 注意事项 ***
    1. 在段落的结尾，请务必补充一段官方声明。eg：<事故调查组按照“实事求是、客观公正、尊重科学”的原则，...,  ... ,现将有关情况报告如下>
    2. 事件描述 简短扼要。
请确保您的文本不仅简要记录了事故概要，而且语言表达准确、连贯，体现出调查报告的权威性和专业性。字数要求 300 - 500 之间。
"""


# Prompt for taking on "data_preprocess" tasks
SECOND_PROMPT = """请根据大语言模型固有知识和常识，详细描述事故发生单位的基本情况以及事故相关人员的情况。请严格按照参考模板的格式返回。
参考模板：
```
一、事故的基本情况
   （一）事故发生单位的基本情况。
        <事故发生单位名称>于<成立年份>年<成立月份>月<成立日期>日成立，法定代表人为<法定代表人姓名>，社会统一信用代码：<信用代码>。公司主要从事<主营业务>，注册地址位于<注册地址>，公司规模<员工人数>人左右, 等等。
   （二）事故相关人员情况。
        <人员详细信息> 
```    
*** 注意事项 ***
    1. 生成文本只包含参考模板里‘事故的基本情况’的章节，不要引入非必要的其他章节。   
"""

# Prompt for taking on "feature_engineering" tasks
THIRD_PROMPT = """请依据“info”所提供的信息，依照以下参考模板，详细叙述事故的发生过程和采取的紧急应对措施。请确保只提供“事故发生经过和事故救援情况”章节的内容，不要包含其他非相关章节。
参考模板：
```
二、事故发生经过和事故救援情况
     <在此处详细描述事故的起因、发展、影响以及救援行动的详细过程，包括事故的评估和救援措施的有效性>
```    
*** 注意事项 ***
    1. 生成文本只包含参考模板里‘事故发生经过和事故救援情况’的章节，不要引入非必要的其他章节。
"""

# Prompt for taking on "model_train" tasks
FIFTH_PROMPT = """
请根据’info’参考信息，结合提供的参考模板，详细描述事故发生的原因和事故性质。请严格按照参考模板的格式返回，并确保每个部分都有合理推理的详细描述。
参考模板：
```
三、事故发生的原因和事故性质
   （一）直接原因。
        <合理推理出发生事故的直接原因，并给出详细的解释和描述>
   （二）间接原因。  
        <合理推理出发生事故的间接原因，并给出详细的解释和描述>
   （三）事故性质。 
        <给出事故性质的明确描述,例如:经事故调查组认定, XXX 事故是一起 XXX 而引发的 xxx 事故。>
```    
*** 注意事项 ***
    1. 生成文本只包含参考模板里‘事故发生的原因和事故性质’的章节，不要引入非必要的其他章节。
    2. 在描述直接原因和间接原因时，请确保推理过程的逻辑性和合理性。
"""


# Prompt for taking on "model_evaluate" tasks
SIX_PROMPT = """请根据’info’参考信息，结合提供的参考模板，详细描述事故责任的认定及处理建议。在描述过程中，需要结合事故发生的具体情况，对责任者的行为与事故发生原因的联系进行深入分析，并根据责任者对事故后果的影响程度进行责任认定。同时，请提供具体的处理建议，包括但不限于刑事责任追究和单位问责。
参考模板：
```
四、事故责任的认定及处理建议
    通过调查事故的经过和分析事故发生的原因，根据责任者的行为与事故发生原因的联系以及对事故后果所起的作用的程度，对事故责任认定如下：
   （一）<对责任者A的行为与事故发生原因的联系进行详细描述，并分析其对事故后果的影响程度，给出责任认定>
   （二）<对责任者B的行为与事故发生原因的联系进行详细描述，并分析其对事故后果的影响程度，给出责任认定>
    对事故责任者及责任单位的处理建议：
   （一）<建议追究刑事责任人员，包括具体的人员名单、职位、建议的刑事责任类型及其依据>
   （二）<问责单位建议，包括具体的单位名称、建议的问责措施及其依据>
```    
*** 注意事项 ***
    1. 生成文本只包含参考模板里‘事故责任的认定及处理建议’的章节，不要引入非必要的其他章节。
    2. 在认定责任时，要确保描述清晰、逻辑严谨，并基于事实和法律规定进行推理。
    3. 在提出处理建议时，要确保建议是可行的、合理的，并且符合相关法律法规的要求。
"""

# Prompt for taking on "image2webpage" tasks
LAST_PROMPT = """请根据“info”中提供的信息，按照以下参考模板，详细列出针对事故的整改及防范措施。请仅关注和填写“事故整改及防范措施”章节，无需涉及其他部分。
参考模板：
```
五、事故整改及防范措施
各县（市、区）必须深刻吸取事故教训，针对事故暴露的问题，进行全面审查和整改。结合本地实际情况，采取切实有效的措施，确保不再发生类似事故。
    （一）<在此填写具体的第一条整改及防范措施>
    （二）<在此填写具体的第二条整改及防范措施>
    （三）<在此填写具体的第三条整改及防范措施>
     ...
```  
"""