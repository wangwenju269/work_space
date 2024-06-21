

#                                   Metagpt 长文档智能写作

+ 贡献：

   本项目代码已贡献到`METAGPT`开源项目里，得到`DeepWisdom` 创始人吴承霖大佬的肯定，本人也将努力为开源社区做贡献。

+ 想法：

   将开源大语言模型和 `Metagpt`框架相互结合，实现智能写作工作。为验证技术路线的可行性，以“事故报告”做实验对象，后续将拓展到论文写作，或者项目招投标(文档级)写作上。

+ 特点：

  对 原  `Metagpt` 结构 未作任何改变， 复用源基层代码，  模仿`datainterpreter` 类进行改造，实现与源代码 高耦合。

+ 技术路线：

  ```mermaid
  flowchart TD
      subgraph 长文档写作
         a(文档摘要) -.-> A
         A[写作大纲] -.大模型任务规划.-> B[Planner]
         a -.人工设计SOP工作流.-> B[Planner]
      end   
      
      subgraph 任务调度拓扑结构
         B -.- 子任务1
         子任务1 -.-> 子任务2 
         子任务2 -.-> 子任务4
         子任务1 -.-> 子任务3 
         子任务1 -.-> 子任务4
         子任务3 -.-> 子任务4 
      end
  
  ```

  + step1 : 首先做任务调度，将复杂的任务拆成若干个小计划。可以人工设定SOP工作流，也可以让模型产生(需要做约束，每个子任务需要符合预先设定的`TaskType` 类型)。

    ```python
    [
        {"task_id": "1", "dependent_task_ids": [], "instruction": "基于用户提供的材料，写背景和意义。", "task_type": "first_paragraph"}, 
        {"task_id": "2", "dependent_task_ids": ["1"], "instruction": "xxx。", "task_type": "second_paragraph"},
        {"task_id": "3", "dependent_task_ids": ["1"], "instruction": "xxx。", "task_type": "third_paragraph"},
        {"task_id": "4", "dependent_task_ids": ["3"], "instruction": "xxx。", "task_type": "forth_paragraph"}, 
        {"task_id": "5", "dependent_task_ids": ["3", "4"], "instruction": "xxx。", "task_type": "fifth_paragraph"}, 
        {"task_id": "6", "dependent_task_ids": ["1", "2", "3", "4", "5"], "instruction": "xxx。", "task_type": "last_paragraph"}
    ]
    ```

    + step2 :  针对每个子任务，执行下述操作。首先大模型根据用户提供的参考信息，先初步定下初稿，生成的初稿让大模型去评价，评价维度如下。后根据评价反馈改进当前初稿。

      ~~~python
      """
      评估维度如下：
          1.语言质量：评估生成的文本是否通顺、语法正确、表达清晰，以及是否符合预期的主题和风格。
          2.内容准确性：评估生成的文本是否包含准确、全面的信息，以及是否能够准确地回答问题或完成任务。
          3.逻辑性：评估生成的文本是否具有合理的逻辑关系等。
          4.上下文相关性：评估生成的文本是否与给定的上下文信息相关，以及是否能够根据上下文信息进行恰当的推理和推断。
          5.安全性：评估生成的文本是否包含敏感信息或不适宜的内容，以及是否能够避免产生歧视性、攻击性或恶意的言论。
      
       Output a list of jsons following the format:
           ```json
          [
              {{
                  "Evaluation_point": str = "依据五个评价标准，按顺序对 report 中的文本内容进行详细评估",
                  "score":int =  "评分，范围在 0-5 之间，用于量化文本质量",
                  "reason": str = "评分的依据和解释",
                  "critique": str = "提供针对生成文本的具体改进建议"
              }},
              ...
          ]
          ```
      """
      ~~~

      ```mermaid
      flowchart TD
          subgraph 子任务
             a(current_task) .-> 写初稿
             写初稿 .-> B((评估))
             B -.通过.-> next_task
             B -.不通过.-> refine改进
             refine改进 -.再次评估.-> B
             refine改进 -.一直评估不通过.-> a
          end   
          
      
      
      ```

+ 缺点：

   目前 还未支持`联网`，`文献检索功能` ,后续会加入。 

+ 目录说明： 

​              data文件：`info.json`用户附带文档信息。`sop.json `是人为设定的`sop` 范式(可选)。

​              user_develop文件 ： 用户自定义开发代码，只需放在`metagpt`源码仓库文件下，即可执行。

+ 开发时间：

​        `metagpt` 源码 大致两周，二次开发时间 3 天

 





























.
