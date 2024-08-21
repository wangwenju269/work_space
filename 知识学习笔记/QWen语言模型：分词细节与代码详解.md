## 代码角度看QWen语言模型：分词细节与代码详解



重点介绍 `AutoTokenizer` 类中 `from_pretrained` 的方法。

**step1 : let's see whether the tokenizer_type is passed so that we can leverage it**

 查看所使用的分词器是否在预设的变量 `TOKENIZER_MAPPING_NAMES` 里。

![](./assets/qwen/image-20231122184948035.png)

**step2 :  let's try to use the tokenizer_config file to get the tokenizer class.**

  step1 中找不到，将加载`tokenizer_config ` 配置文件

```
resolved_config_file = cached_file()  # 尝试在本地文件夹和存储库中查找文件，必要时下载并缓存它。
 ...
 # 加载 tokenizer_config.json 里文件
  with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
```

接着执行会得到两个重要的变量，` config_tokenizer_class` =   'QWenTokenizer',  ` class_ref`= 'tokenization_qwen.QWenTokenizer'

```python
 # 获取动态的 QWenTokenizer 类模块
 tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
 """
 该函数嵌套 get_class_in_module函数，get_cached_module_file(class_name, final_module.replace(".py", ""))
 get_cached_module_file：Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached  Transformers module.
 
 其中class_name = 'QWenTokenizer'  final_module = 'transformers_modules/Qwen-14B-Chat/tokenization_qwen.py'
 可以理解将本地local下的脚本文件（Qwen-14B-Chat）上传到 远程 transformers 模块下，即dynamic_module_path，便于使用tansformers其他基本组件模块。
 dynamic_module_path ： .cache/huggingface/modules/transformers_modules/Qwen-14B-Chat
 """
```

​    看明白以上操作后，本质是执行这行代码：

```python
tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
"""
可以 看下 tokenizer_class 的类型
type(tokenizer_class): <class 'transformers_modules.Qwen-14B-Chat.tokenization_qwen.QWenTokenizer'>
而QWenTokenizer继承的是 PreTrainedTokenizer， 执行的是PreTrainedTokenizer(PreTrainedTokenizerBase).from_pretrained.  
直至现在，你终于理解AutoTokenizer.from_pretrained（trust_remote_code = True）， trust_remote_code 这个参数的意义，就是本地local下的.py
文件复制到 远程transformers模块下。
"""
>>> # Instantiate the tokenizer.
>>> try:
>>>    tokenizer = cls(*init_inputs, **init_kwargs)
```



简单介绍一下，`AutoModelForCausalLM`

`AutoModelForCausalLM`  继承  `_BaseAutoModelClass` 基类，在文件 `auto_factory` 文件中。 
同样的原理，同样的套路

```python
model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs)
'''
type(model_class): <class 'transformers_modules.Qwen-14B-Chat.modeling_qwen.QWenLMHeadModel'>
'''
model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)

```

大致查看下文件`model.safetensors.index.json` 所保存的文件信息，就是权重文件与图结构的映射关系图。

![image-20231127173818687](./assets/qwen/image5.png)

接着进入上下文管理器，实例化模型结构图

```
with ContextManagers(init_contexts):
     model = cls(config, *model_args, **model_kwargs)
```

加载权重

```python
cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ...)

# 本质上执行的是 load_state_dict 这个函数
...
...
...
def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
            if metadata.get("format") not in ["pt", "tf", "flax"]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
                return safe_load_file(checkpoint_file)
```

transformers中的generate函数解析工作的介绍

```python
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        
"""
参数：
inputs (torch.Tensor of varying shape depending on the modality，optional):
生成使用的序列或模型输入到编码器。如果None，方法将它初始化为bos_token_id和一个大小为1的批次大小。对于只包含解码器的模型，inputs应该以input_ids的形式输入。对于编码器-解码器模型，inputs可以代表input_ids，input_values，input_features或pixel_values的任何一种。
generation_config (~generation.GenerationConfig，optional):
用于生成的基参数化。如果generation_config不可用，则默认值将使用模型配置中的默认值。如果提供的参数与generation_config中的参数匹配，则将使用这些参数。如果不提供generation_config，则将使用以下加载顺序：1）从generation_config.json模型文件中获取；2）从模型配置中获取。请注意，未指定的参数将继承~generation.GenerationConfig的默认值，其文档应该用于参数化生成。
logits_processor (LogitsProcessorList，optional):
用于补充默认logits处理器的自定义logits处理器。如果提供的logits处理器已经使用了相同的参数或生成配置，则会引发错误。此功能旨在为高级用户提供便利。
stopping_criteria (StoppingCriteriaList，optional):
用于补充默认停止准则的自定义停止准则。如果提供的停止准则已经使用了相同的参数或生成配置，则会引发错误。此功能旨在为高级用户提供便利。
prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]，optional):
如果提供，则此函数仅约束搜索到的令牌。如果未提供，则不应用任何约束。此函数需要两个参数：批次IDbatch_id和input_ids。它应该返回一个条件为batch_id和以前生成的令牌inputs_ids的令牌列表。此功能可用于约束带前缀的生成，如自回归实体检索中所述。
synced_gpus (bool，*optional，默认为False):
是否继续运行循环直到最大长度（需要ZeRO阶段3）
kwargs：
随机参数化generate_config和/或特定于模型的
"""        
```

 **参考链接  [解码](https://www.likecs.com/show-308663700.html)**

