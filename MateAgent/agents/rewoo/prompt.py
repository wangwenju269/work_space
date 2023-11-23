PLANNER_PROMPT_CN = """Task decomposition is often necessary to efficiently address user questions. However, in certain cases, it may not be required. Consider the following factors to determine if task decomposition is necessary:Task decomposition is often necessary to efficiently address user questions. However, in certain cases, it may not be required. Consider the following factors to determine if task decomposition is necessary:
1. Question simplicity: If the question is about common sense, simple world knowledge, or doesn't involve multiple subtasks, a direct reply is sufficient.     
2. Ability to provide a direct answer: If you have the necessary information to directly address the user's question, a direct reply is appropriate.
3. Avoiding unnecessary operations: If additional operations or steps won't contribute to a more accurate or comprehensive response, it's best to skip task decomposition and promptly provide the appropriate reply.

If the question satisfies these conditions, you can provide a direct reply without task decomposition. However, for tasks that require decomposition, follow the program logic and break down the question into subtasks to gather the necessary information. Please structure your response in the following format:
```
subtasks: the problem to be solved by the current subtask
#E[id] = tool_name(tool_parameters)
... (this subtasks/#E[id] can be repeated one or more times,you should Minimize the number of subtasks.)

```
*** NOTE ***
1. #E[id] is used to store the execution result of the subtasks and can be used as a placeholder.
2. Each plan should be followed by only one #E.
3. The content implemented by each #E[id] should strictly correspond to the problem currently planned to be solved.
4. Tool parameters can be entered as normal text, or #E[dependency_id], or both.
5. Make sure to use the tools reasonably and correctly, selecting the appropriate tool from the provided options:
{tool_description}


Remember the following important guidelines:
```
- Never create new subtasks that similar or same as the exisiting subtasks.
- Do not make irrelevant or unnecessary subtasks.
- Make sure your subtasks can fully utilze its ability and reduce the complexity.
```

为方便你理解问题,提供了一些案例你可以学习。
Question: 李白是谁？
Return : 李白是诗人 

Question: What is Leo Dicaprio's girlfriend's age to the power of 0.34? 
subtasks: Find out the name of Leo Dicaprio's girlfriend.
#E[1] = search(query="name of Leo Dicaprio's girlfriend")
subtasks: Find out the age of Leo Dicaprio's girlfriend.
#E[2] = search(query="age of #E[1]")
subtasks: Calculate her age to the power of 0.34.
#E[3] = math(query="#E[2]^0.34")

Question:绘制一幅包含《咏鹅》诗意信息的图片。
subtasks: 搜索《咏鹅》古诗的内容信息
#E[1] = search(query="咏鹅古诗")
subtasks: 绘制一幅包含诗意的图片
#E[2] = image_gen(prompt="A picture of a swan singing in a lake.")
(END OF examples)

Question:{query}
"""

WORKER_PROMPT_CN = """
思考: {thought}\n回答: {action_resp}
"""
SOLVER_PROMPT_CN = """为了帮助你解决任务,我们提供了与任务相关的信息。
注意: 
1. 其中一些信息可能存在噪声,你需要谨慎的使用它们。
2. 如果这个问题的答案已经在信息中存在,直接提取,返回精炼的答案即可。
信息:
```
{worker_log}
```

请结合上文信息来回答问题,答案要总结归纳,精炼准确。
问题:{question}
"""












