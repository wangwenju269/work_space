CHECK_PROMPT = """针对当前问题,你需要依据工具描述和外部知识, 返回可供工具解析的正确参数以解决当前问题。按照以下格式返回：
```
当前问题: 当前要解决的问题
外部知识: 解决当前问题所需的外部知识
答案: 函数名(待解析参数)
```
注意: 工具使用描述如下
{tool_description}
为方便你理解问题,提供了一些案例你可以学习。\n
{EXAMPLES}
当前问题: {cur_question}
外部知识: {history}
答案:
"""

EXAMPLE_math = """当前问题: 今年盈利收益比去年盈利收益高出多少?
外部知识:
1. 今年盈利收益为52500。-->去年盈利收益为25000
答案: math(query = "(52500-25000)")
""" 
EXAMPLE_speech_synthesis = """当前问题: 朗读《出塞》古诗
外部知识:
1. 依旧是秦时的明月汉时的边关,征战长久延续万里征夫不回还。倘若龙城的飞将李广而今健在,绝不许匈奴南下牧马度过阴山。
答案: speech_synthesis(prompt = "秦时明月汉时关,万里长征人未还。但使龙城飞将在, 不教胡马度阴山。")
""" 


EXAMPLE = {'math':EXAMPLE_math,
           'python': '' ,
           'search': '' ,
           'speech_synthesis' : EXAMPLE_speech_synthesis
          }