REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:
{tool_descs}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)


Final Answer: the final answer can successfully solve the original question.

{reflection}
Begin! 
Question: {query}"""




REFLECTION_HEADER = """*** Important Notice ***
The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.
"""

REFLEXION_REACT_PROMPT = """
You are an advanced reasoning agent that can improve based on self refection.You will be given a previous reasoning trial in which you were given access access to the following tools.
{tool_descs}

You were unsuccessful in answering the question due to one of the following reasons:
- Incorrect Thought of Reasoning.
- Missing intermediate reasoning steps
- The Thought of Reasoning requires further refinement to reach a solution.
- Mistakenly invoking an incorrect Tool API.
- Error in parameter parsing of the Tool API.

In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure.

Here are some examples:

Previous Trial:
Question: 《罗马议定书》是由哪几位首相签署的？请问被暗杀的是哪一位首相？
Thought: 请帮我搜索《罗马议定书》，找到其中的三位首相。
Action: search
Action Input: {{"query": "《罗马议定书》"}}
Observation: 《罗马议定书》是由意大利总理贝尼托·墨索里尼、奥地利总理恩格尔伯特·多尔福斯和匈牙利总理久拉·戈姆博这三位首相签署的。
Thought: 逐一搜索三位首相，确定哪位首相遭受暗杀。
Action: search
Action Input: {{"query": "墨索里尼、多尔福斯、戈姆博签署"}}
Observation: Answer is INCORRECT
Reflection: 顺序逐一搜索每位首相的信息是我的应行做法，不应该同时搜索三位首相。我应该按照代码程序的思维进行逐一搜索。


Question: 请问广东与哈尔滨的气温相差多少摄氏度？
Thought: 我需要获取广东和哈尔滨的实时气温，然后计算它们之间的温差。
Action: weather_api
Action Input: {{"location": "广东,哈尔滨"}}
Observation: Answer is INCORRECT
Reflection: The tool usage is incorrect,The "location" parameter should be a single city name, not a combination of two cities, I should use the weather API to get the current temperature of each city separately and then calculate the temperature difference.

Previous Trial:
Question: 计算百度总裁年龄的五次幂减去华为首席执行官年龄的三次方,该数值是多少? 
Thought: 我需要使用math API来计算这个数学问题。
Action: math
Action Input: {{"query": "((age_baidu_chief)^5) - ((age_huawei_chief)^3)"}}
Observation: Answer is INCORRECT
Reflection: 由于我没有获得百度总裁和华为首席执行官的年龄，所以我无法计算这个数学问题的答案。我应该首先使用search API来获取这些信息，然后在使用math API来计算答案。
(END OF examples)

Begin! 
Previous Trial:
Question:{query}
Reflection:"""