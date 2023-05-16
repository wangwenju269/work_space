# PyTorch_BERT_Pipeline_IE

# 概述
1、复习论文《A Frustratingly Easy Approach for Joint Entity and Relation Extraction》<br>
2、模型的具体思路描述可以见[知乎](https://zhuanlan.zhihu.com/p/369951155)。<br>
3、训练数据来自于百度的[2021LIC语言与计算智能竞赛](https://aistudio.baidu.com/aistudio/competition/detail/65)，没有上传数据，可以自行到官网下载哦。<br>

# 环境要求
```
pytorch >=1.6.0
transformers>=3.4.0
```
# 运行步骤
1、进入entity_extract项目或relationship_classifiction项<br>
2、去huggingface[官网](https://huggingface.co/models)下载BERT预训练权重，然后并放在`./pretrained_model/`文件夹下<br>
3、在`./utils/arguments_parse.py`中修改BERT预训练模型的路径<br>
4、运行`train.py`进行训练<br>
5、运行`predict.py`进行预测
