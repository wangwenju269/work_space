PLANNER_PROMPT_CN = """请将 Question 拆分为(1个或2个)子任务,从而能够得到充分的信息以解决问题, 按照以下格式返回：
```
Plan: 当前子任务要解决的问题
#E[id] = 工具名称[工具参数]
Plan: 当前子任务要解决的问题
#E[id] = 工具名称[工具参数]
```
其中
1. #E[id] 用于存储Plan id的执行结果, 可被用作占位符。
2. 每个 #E[id] 所执行的内容应与当前Plan解决的问题严格对应。
3. 工具参数可以是正常输入text, 或是 #E[依赖的索引], 或是两者都可以。
4. 工具名称从一下工具中选择：
{tool_description}
注意：输出不要重复,拆分不允许超过3个子任务。
为方便你理解问题,提供了一些案例你可以学习。

Question:思考《咏鹅》古诗所表达的思想是什么？
Plan: 搜索《咏鹅》古诗的内容信息
#E[1] = search(query="咏鹅古诗")

Question:绘制一幅包含《咏鹅》诗意信息的图片。
Plan: 搜索《咏鹅》古诗的内容信息
#E[1] = search(query="咏鹅古诗")
Plan: 绘制一幅包含诗意的图片
#E[2] = image_gen(prompt="A picture of a swan singing in a lake with the words '咏鹅' written on it.")

Question:{query}
"""

WORKER_PROMPT_CN = """
想法: {thought}\n回答: {action_resp}
"""
SOLVER_PROMPT_CN = """为了帮助你解决任务,我们提供了与任务相关的信息。
注意: 其中一些信息可能存在噪声，因此你需要谨慎的使用它们。
信息:{worker_log}
请回答下面问题,答案要总结归纳,精炼准确,不要包含其他不需要的文字。
问题:{question}
"""




INTENT: str = """现在有一些工具,名称为{intents}。
你的任务是理解用户问题的意图，并判断该问题使用哪几种工具。
工具名称从一下工具中选择：
{tool_description}
注意：必须按格式回复：“工具名称：<>”。

举例：
问题：今天的天气怎么样？
工具名称：<weather_api,search>

问题：画一幅内容为山水鸟虫的画。
工具名称：<search,image_gen>

问题：将下面的文字转成语音：<文本>
工具名称：<speech_synthesis>

问题：“{query}”
"""