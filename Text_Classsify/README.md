# Classify_BERT

*Author : wangwenju；QQ: 2413887576； 邮箱：wangwenju_wangs@163.com*

***

**specific task: 多类别文本分类任务**

***

+ <u>**代码说明**</u>
  + **train** ：启动模型训练流程，包括数据和模型加载，优化器、损失函数设置等。
  + **data_pre_process** ： 将 .json 数据文件读取，输入Data loader，提供两种创建方法。
  + **pretrained** :  BERT预训练模型的路径，本实验采用 chinese_roberta_wwm_ext。包含三个必要文件：(config.json,  vocab.txt, pytorch_model.bin), 可以在hugging face 官网下载，并放置在指定参数路径下。
  + **bert2classify**： 加载BERT模型框架，BERT输出可以接具体任务组件。
  + **metrics**  : 评估指标,根据具体任务来设置，多分类评估采用ACC，宏平均F1，Recall 等。
  + **utils.logger** ：日志记录文档，方便调试。
  + **output** ： 模型参数输出的保存文件。



+ <u>**参考链接**</u>

  +  [huggingface 官网教程](https://huggingface.co/course)

  + [杨夕](https://github.com/km1994/nlp_paper_study)

    
