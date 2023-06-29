## <center>基于UIE-X模型跨长文档级别的信息抽取案例教程 </center>



+ 本项目采用UIE-X模型实现跨文档级别信息抽取，在少样本上进行微调，实现文档级别的端到端应用方案，打通长文档格式处理-数据标注-模型训练-模型调优-预测部署全流程，可快速实现文档信息抽取产品落地。

### 跨文档处理

1. 业务场景下的数据大多为非结构化的数据,如word文档(.docx)、excle文档(.xlsx)、PDF文档、jpg图片等。采用`UIE-X`模型需要统一化处理这些格式化形式。

2. `langchain`框架集成多种非结构化文档处理形式，推荐使用 `langchain`

```
# 加载langchain UnstructuredFileLoader 函数
from langchain.document_loaders import UnstructuredFileLoader
"""
TXT文档载入导入文本
"""
loader = UnstructuredFileLoader("./example_data/state_of_the_union.txt")
document = loader.load()
print(docs[0].page_content)
"""
DOCX文档读入(word)
"""
loader = UnstructuredWordDocumentLoader("example_data/fake.docx")
data = loader.load()
```

```python
# 将长文本切分若干个segment篇落
from langchain.text_splitter import RecursiveCharacterTextSplitter
"实例化对象:每个篇落 chunk_size 字符，片落前后重叠为 chunk_overlop"
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 100)
"document 是 Document实例化对象"
split_document = text_splitter.split_documents(document)
"document 是字符串"
split_document = text_splitter.create_documents([document])
# 遍历文档：
for d  in split_document:
    print(d.page_content)
```

### 数据正则模式标注

1.`UIEX`模型依赖百度Label_Studio标注平台，会生成`label_studio.json`文件，经`DataConverter`转化成训练集和验证集。

2.针对用户私有化的数据，为避免数据泄露，自己提出一种数据标注方案，完全不用依赖label_studio。

```python
1. 在公开数据集tax上分析label_studio平台的标注模式

tax_data_file = './tax/train.txt'
tax_data = open(tax_data_file,'r' encoding = 'utf-8').read()
tax_data = tax_data.strip().split('\n')
# 分析某一个样本   tax_one  = eval(tax_data[0])
# tax_one 字典形式存放，keys 有 content \ result_list \ prompt \ image\ bbox
# 查看result_list 键值; 存放着 text / start / end 


2.标注模式：    
      UIEX模型采用 pointer_based 标注模式，既标注待提取短语的首位起始位置；
      UIEX模型并不是论文上所述的方法（text2structured）,是 基座模型是ERNIE_layout + mrc 方法            

      构造标注的难点：
         在于怎么准确找到start end 的 ids 信息，查看paddle源码，只需找到 target 分词id 在文本中的id 的位置信息即可。


3. 设计自己的标注方法：

   标注样例：
    
   content：str              #   输入word 文档 转换 长字符串篇落
   
   label：{ prompt ： target }    #   prompt 提示短语  target 待提取的标注短语，target短语在content文本中出现，
                            target 提取短语句子较长，且存在控制符、换行符等，通过正则的方法进行标注。
                                   
   

eg： 采用正则模式进行标注，减少target的格式与word 文档里 content 不一致现象；
content = '小明的公民身份证号：[99322], 年龄：ddd '
label = {'身份证号'："[./d*?]"}  
# label形式为： {prompt ： target}
import re
def search(pattern,sequences):
     target = re.search(pattern,sequences)
     if target:
        return target.group(0)
     else:
        return -1
# 通过这种方法，就可以将文档中出现的target 的跨度提取出来
example = {'content' : content, 'result_list':[{'text': 99322, 'start': None ,'end' : None }],"prompt" : '身份证号'}



4. 构造UIEX模型的输入形式
   正则的方法将target的内容取出来，保存在example["result_list"][0]['text']里 ，
   接着需要确定 result_list 里 start 、end 首尾位置 id 位置信息；

encoded_inputs = tokenizer(
                 text = [example['prompt']],
                 text_pair = [example['content']],
                 ...,
                 ...
)
# UIEX 输入形式： cls  prompt  sep  sep content sep
input_ids = encoded_inputs['input_ids']
# 找 text 文本分词后在 input_ids 所对应的首尾位置id
text_ids = tokenizer(text)[1:-1]
def search_id(text_ids,input_ids):
    n = len(text_ids)
    for i in range(len(input_ids):
        if input_ids[i:i + n ] == text_ids:
                   return i 
    else:
                   return -1
    
start = search_id(text_ids,input_ids)
if start != -1:                   
    end = start + len(text_ids) - 1
# 这样就能得到 start end id信息，填入上述example['result_list'] ，就可以完成构造样本。
```

#### 总结：
```python
  1. 正则模式标注待提取target的span跨度；
  2. 分别对文档和target 进行分词处理，文档：input_ids， target：text_ids 
  3.  text_ids 在 input_ids 序列中，确定 start end 的首尾 id值 
  4. 转换UIEX模型所需的样本example形式；
```

### 模型微调

1. 官方tax案例微调代码：https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/informationextraction/document

****

### 推理测试

```python
1. 第一种方案：
   Taskflow ： 当schema待提取数量集合过多时，调用 Taskflow 运行性能低，很耗时；
2. 第二种方案： 调用 底座模型 ERNIE_layout 模型
   model = UIEX.from_pretrained(checkpoint)
   for batch in infer_data_loader:
      start_prob , end_prob = model(**batch)  
   实验证实：直接调用底座模型，推理速度更快
```

