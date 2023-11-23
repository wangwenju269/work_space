_PROMPT_TEMPLATE = """Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.

Question: ${{Question with math problem.}}

```text
${{single line mathematical expression that solves the problem}}
```

...numexpr.evaluate(text)...

```output
${{Output of running the code}}
```

Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?

```text
37593 * 67
```

...numexpr.evaluate("37593 * 67")...

```output
2518731
```

Answer: 2518731

Question: {question}
"""
import math
import numexpr
import re


class Math:
    def __init__(self,llm) :
        self.llm = llm
        self.im_start = '<|im_start|>'
        self.im_end =  '<|im_end|>'
        self.prefix = f'{self.im_start}system\nYou are a helpful calculator assistant.{self.im_end}'
        
    def construct_prompt(self,query):
        im_start, im_end, prompt = self.im_start,  self.im_end, self.prefix
        query = _PROMPT_TEMPLATE.format(question = query)
        query = query.lstrip('\n').rstrip()   
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{im_end}"
        assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
        prompt = prompt[: -len(f'{im_end}')]
        return prompt


    def _process_llm_result(self, llm_output: str) :
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return answer

    def _evaluate_expression(self, expression: str) -> str:
        try:
            local_dict = {"pi": math.pi, "e": math.e}
            output = str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except Exception as e:
            raise ValueError(f"{e}. Please try again with a valid numerical expression")

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    def run(self,task):
        prompt = self.construct_prompt(task)
        respones = self.llm.text_completion(prompt,stop_words = ['output'] )
        answer = self._process_llm_result(respones)
        return answer 

if __name__ == '__main__':

    import sys
    sys.path.append('MateAgent')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from config.parser import DataArguments
    from llm.model import Qwen  
    args = DataArguments()
    qwen = Qwen(args.checkpoint)
    task = '20^6'
    math_ = Math(llm = qwen)
    answer = math_.run(task)
    print(answer)

    #  text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)

