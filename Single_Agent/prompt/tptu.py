PLANNER_PROMPT_CN = """请将 Question 拆分为 (1个或2个) 子任务,从而能够得到充分的信息以解决问题, 按照以下格式返回：
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
4. 工具名称必须从一下工具中选择：
{tool_description}
注意：输出不要重复,子任务 Plan 也可以不使用工具。 
为方便你理解问题,提供了一个案例你可以学习。

Question:搜索《咏鹅》古诗的内容信息，绘制一幅包含诗意的图片。
Plan: 使用搜索工具搜索《咏鹅》古诗的内容信息
#E[1] = google_search(query="咏鹅古诗")
Plan: 绘制一幅包含诗意的图片
#E[2] = image_gen(prompt="A picture of a swan singing in a lake with the words '咏鹅' written on it.")

Question:{query}
"""


WORKER_PROMPT_CN = """
想法: {thought}\n回答: {action_resp}\n
"""

SOLVER_PROMPT_CN = """为了帮助你解决任务,我们提供了与任务相关的信息。
注意其中一些信息可能存在噪声，因此你需要谨慎的使用它们。
{question}
信息：{worker_log}
现在开始回答这个任务。请直接回答这个问题,不要包含其他不需要的文字,回答问题精炼准确。
{question}
"""
