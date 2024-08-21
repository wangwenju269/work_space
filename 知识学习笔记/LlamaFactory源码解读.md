



#                                   微调

启动命令：

```shell

deepspeed --include="localhost:2,3,6" --master_port 2411  LLaMA-Factory/src/train.py 
    --deepspeed   workspace/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --stage          pt \
    --do_train  \
    --do_eval  \
    --val_size               0.2  \
    --eval_steps             10  \
    --evaluation_strategy    steps  \
    --dataset               gongwen_writer  \
    --model_name_or_path    Qwen-14B-Chat  \
    --finetuning_type   freeze  \
    --num_layer_trainable  1   \
    --template          qwen  \
    --per_device_train_batch_size  2  \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps  2  \
    --lr_scheduler_type cosine  \
    --logging_steps    10  \
    --learning_rate    1e-5   \
    --plot_loss  \
    --gradient_checkpointing  \
    --save_strategy          epoch  \
    --num_train_epochs       4  \
    --cutoff_len 8192    \
    --bf16  \
    --output_dir    /WriteReport/checkpoint/pt2 \
    --dataset_dir   /LLaMA-Factory/data  
    
deepspeed --include="localhost:1,2,3,5"   /LLaMA-Factory/src/train.py \
    --deepspeed    \
    --stage          sft \
    --do_train      \
    --do_eval       \
    --val_size               0.2  \
    --eval_steps             10  \
    --evaluation_strategy    steps  \
    --dataset               id \
    --model_name_or_path       checkpoint/pt/checkpoint-12   \
    --finetuning_type        lora        \
    --lora_rank 8  \
    --lora_target       c_proj,c_attn    \
    --template          qwen    \
    --per_device_train_batch_size  4  \
    --per_device_eval_batch_size 2   \
    --gradient_accumulation_steps  4  \
    --lr_scheduler_type cosine  \
    --logging_steps    10  \
    --learning_rate    1e-5  \
    --plot_loss  \
    --gradient_checkpointing  \
    --save_strategy          epoch  \
    --num_train_epochs       8  \
    --cutoff_len  2048     \
    --overwrite_cache       \
    --overwrite_output_dir  \
    --output_dir        \
    --dataset_dir   /LLaMA-Factory/data  
      
  
CUDA_VISIBLE_DEVICES=1  python src/api.py     --model_name_or_path  checkpoint/export  --template qwen    
  
curl -X POST http://0.0.0.0:8000/v1/chat/completions -H "content-type:application/json" -d '{
  "messages":[{"role":"user","content":"xinqingbuhao,请将拼音转化汉字"}],
  "model": "qwen",
  "stream": false,
  "max_tokens": 256
}'
 
python src/export_model.py  \
    --model_name_or_path    checkpoint/pt/checkpoint-12  \
    --adapter_name_or_path  checkpoint/sft_last/checkpoint-51  \
    --template qwen  \
    --finetuning_type lora \
    --export_dir  \
    --export_size 2 \
    --export_legacy_format False            
```



下面逐行调试的方法，细节来看SFT代码：

~~~mermaid
graph LR
  
    a((run_sft)) --> b[load_tokenizer]
    a --> c[get_dataset]
    a --> d[load_model]
    c --> c_1[get_dataset_list]
    c_1 --> c_2[load_single_dataset]
    c_2 --> c_3[get_preprocess_and_print_func]
    d  -->|AutoModelForCausalLM| d_1[from_pretrained]
    d_1 --> d_2[register_autoclass]
    d_2 --> |peft| d_3[get_peft_model]
    b -->|AutoTokenizer| b_1[from_pretrained]
    d_3 --> model
    style model stroke:#333,stroke-width:4px;
    b_1 --> tokenizer
    style tokenizer stroke:#333,stroke-width:4px;
    c_3 --> dataset
    style dataset stroke:#333,stroke-width:4px;
    tokenizer --->|DataCollatorForSeq2Seq| x((data_collator))
    model -->  trainer
    tokenizer -->  trainer
    x -->  trainer
    dataset  -->  trainer
~~~



+ #### 准备项： 

  配置参数信息，推荐一种方法直接配置 `YAML`文件，并加载到`sys.argv` 变量里

  ```python
  # 在train_bash.py 文件里顶部添加这两行代码
  import sys
  sys.argv.append("/workspace/LLaMA-Factory/config.yaml") # 这是 yaml 文件路径
  # 也就是把 终端里命令行里的参数信息，直接方法 yaml 文件里，直接 debug train_bash.py 文件
  ```

  解析参数：

  ```python
  # 参数解析代码 在 LLaMA-Factory/src/llmtuner/hparams/parser.py 文件里
  def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
      parser = HfArgumentParser(_TRAIN_ARGS)
      return _parse_args(parser, args)
  """
  HfArgumentParser 是 Hugging Face Transformers 库中的一个类，它用于解析命令行参数和配置文件，以便于在训练和部署基于Transformers的模型时使用。
  HfArgumentParser 继承自 Python 的内置 argparse 模块，提供了对 YAML 配置文件的支持，可以在不同的环境中配置和共享参数。
  _TRAIN_ARGS:[基于 dataclass 修饰的 类配置]
  """
  # 读文件
  parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
  """
  一个替代的辅助方法，完全不使用 `argparse`，而是加载一个yaml文件并填充数据类类型。
  参数:
      yaml_file (`str` or `os.PathLike`):
          要解析的yaml文件的文件名
      allow_extra_keys (`bool`, 可选, 默认为 `False`):
          默认为 False。如果为 False，并且yaml文件中包含未被解析的键，将引发异常。
   返回:
      一个元组，包含：
         - 数据类实例，按照它们传递给初始化器的顺序排列
  """
  ```

    

  仔细阅读代码不难发现：

  ​           `parser.parse_args_into_dataclasses(return_remaining_strings=True)`  同理也可以实现，将命令行参数解析为指定数据类类型的实例。展开内部细看：

  `parse_args_into_dataclasses `函数入参说明

  + args 参数是一个字符串列表，包含了要解析的命令行参数。如果没有提供，默认会使用 sys.argv，这是 Python 中一个包含命令行参数的列表。
  + return_remaining_strings 参数是一个布尔值，如果设置为 True，函数不仅会返回解析后的参数，还会返回一个列表，包含所有未被解析器识别的剩余参数字符串。
  + look_for_args_file 参数是一个布尔值，如果设置为 True，函数会在与当前进程的入口点脚本同名的目录下查找一个名为 “.args” 的文件，并将其内容作为额外的命令行参数添加到 args 中。# 当参数信息也可以保存在 .args 后缀文件里，但要求文件名与启动脚本文件名一致，如 train_bash.args.
  + args_filename 参数是一个字符串，如果被指定，函数将使用这个文件名来代替默认的 “.args” 文件。
  + args_file_flag 参数是一个字符串，如果被指定，函数会在命令行参数中查找这个标志，并使用标志后面的文件作为参数文件。如果命令行中多次出现这个标志，将以最后一次出现的文件为准。

  ```python
  for dtype in self.dataclass_types:
      keys = {f.name for f in dataclasses.fields(dtype) if f.init}
      inputs = {k: v for k, v in vars(namespace).items() if k in keys}
      for k in keys:
          delattr(namespace, k)
          obj = dtype(**inputs)
          outputs.append(obj)
  """
  后续代码: 将所有参数信息，全部加载到 namespace 里， 然后遍历每一个数据类，找到预先定义的参数，并从 namespace 里去除该属性。
  """        
  # namespace 怎么来的，看以下代码，设置参数属性并校验。
  if namespace is None:
      namespace = Namespace()
      for action in self._actions:
          if action.dest is not SUPPRESS:
              if not hasattr(namespace, action.dest):
                  if action.default is not SUPPRESS:
                      setattr(namespace, action.dest, action.default)    
  
  """如果必要参数缺失，直接抛出错误："""                   
  #  train_bash.py: error: the following arguments are required: --model_name_or_path, --output_dir                 
  ```

  

+ #### SFT 核心代码块

  `LLaMA-Factory/src/llmtuner/train/sft/workflow.py`

  - ### ` load_tokenizer  `

    ```python
    tokenizer = AutoTokenizer.from_pretrained(
                            model_args.model_name_or_path,
                            use_fast=model_args.use_fast_tokenizer,
                            split_special_tokens=model_args.split_special_tokens,
                            padding_side="right",
                            **init_kwargs,)	
    patch_tokenizer(tokenizer)
    """
    patch_tokenizer 目的是确保 tokenizer 对象有一个正确实现的 _pad 方法，这个方法用于处理序列填充（padding）。
    如果 tokenizer 对象没有从 PreTrainedTokenizerBase 继承 _pad 方法，确保 tokenizer 可以正确地进行序列填充。
    tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
    """
    ```

  - ### `get_dataset`

    get_dataset_list 函数这部分主要是数据读取和数据转化为`DatasetAttr` 属性，load_single_dataset 从文件中读取数据，并对齐操作。 

    - 文件 `DATA_CONFIG`  也就是 `'dataset_info.json'` （依赖文件）读取到 `dataset_info`里，调试注意相对路径问题，建议`dataset_dir`参数
    -   `column_names.extend(["prompt", "query", "response", "history"])` : dataset_info 里 `columns `字段依次对齐到上述字段。

    ###### 1.`load_dataset`

       这段代码使用 Hugging Face Transformers 库中的 `load_dataset` 函数来加载一个数据集。官方翻译如下：

    ```text
    这个函数在幕后执行以下操作：
    1. 如果库中没有缓存，则从 path 下载并导入数据集脚本。
    
        如果数据集没有数据集脚本，那么会导入一个通用的数据集脚本（JSON, CSV, Parquet, text 等）。
        数据集脚本是小型的 Python 脚本(定义了数据集构建器)。它们定义了数据集的引用、信息和格式，包含了原始数据文件的路径或 URL，以及从加载示例的代码。
    
    2. 运行数据集脚本，这将：
    
        * 如果本地或缓存中尚不可用，从原始 URL（参见脚本）下载数据集文件。
        * 处理并将数据集缓存为类型化的 Arrow 表格以便缓存。Arrow 表格是任意长度的、支持嵌套对象的类型化表格，可以映射到 numpy/pandas/python 通用		类型。它们可以直接从磁盘访问，加载到 RAM 中，甚至可以通过网络流式传输。
    
    3.返回由 split 请求的分割构建的数据集（默认：所有分割）。
    ```

    ```python
    dataset = load_dataset(
                    path=data_path,
                    name=data_name,
                    data_dir=data_dir,
                    data_files=data_files,
                    split=data_args.split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.hf_hub_token,
                    streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
                    **kwargs,
    )
    """
    path: 根据传入的 path，使用的数据集构建器可能来自一个通用的数据集脚本（例如 JSON, CSV, Parquet, text 等）或者来自数据集目录内的数据集脚本（一个 Python 文件）。
    
    对于本地数据集：
            如果 path 是一个本地目录（只包含数据文件） -> 根据目录内容加载一个通用的数据集构建器（csv, json, text 等） 
            例如：'./path/to/directory/with/my/csv/data'。
            如果 path 是一个本地数据集脚本或一个包含本地数据集脚本的目录（如果脚本与目录同名） -> 从数据集脚本加载数据集构建器
            例如：'./dataset/squad' 或 './dataset/squad/squad.py'。
    
    name: 数据集的名称。对于 Hugging Face Hub 上的数据集，这是数据集的版本。对于本地数据集，这个参数通常不需要设置。
    data_dir: 数据集所在的目录。如果指定了 data_dir，load_dataset 将在这个目录中查找数据集。
    data_files: 一个字典或列表，指定了数据集文件的位置。
    split: 要加载的数据集的分割（split）。这可以是 "train"、"validation"、"test" 等，或者是自定义的分割名称。
    cache_dir: 缓存数据集的目录。如果指定了 cache_dir，下载的数据集将被缓存在这个目录中，以便于下次快速加载。
    token: Hugging Face Hub 的访问令牌。如果数据集是私有的或者是需要认证的，可以使用这个参数来提供访问令牌。
    streaming: 是否以流式方式加载数据集。如果设置为 True，数据集将以流式方式加载，这意味着数据集不会被完全加载到内存中，而是按需加载。这对于大型数据集非常有用。
    **kwargs: 其他关键字参数。这些参数将传递给 load_dataset 函数，用于处理特定情况或提供额外的配置选项。
    """
    ```

      load_dataset 具体实施细节底层代码相对复杂，大致过程：

    ```text
    step1：模块加工 --> dataset_module_factory
    step2: 类构建 --> get_dataset_builder_class
    ```

    返回结果如下：

    ```python
    Dataset({
        features: ['history', 'input', 'output', 'instruction'],
        num_rows: 54
    })
    ```

    ######  3.`align_dataset`

    数据对齐操作，转化为问答对形式

    ```python
    Dataset({
        features: ['prompt', 'response', 'system', 'tools'],
        num_rows: 54
    })
    """
    展开来看：
    'prompt': [{'role': 'user', 'content': ''}]
    'response':[{'role': 'assistant', 'content': ''}]
    'system':''
    'tools':''
    """
    ```

    ###### 4.`get_preprocess_and_print_func` 数据处理

       转化模型所需三要素条件，`input_ids`,`attention_mask`,`labels`, 采用移位`multi_label` 方式，预测时，只计算回应的损失。

    ```
    """
    IGNORE_INDEX : -100
    source_ids: prompt   ---tokenzier2ids---> ids 
    target_ids: response
    """
    source_mask = [IGNORE_INDEX] * len(source_ids)
    input_ids += source_ids + target_ids
    labels += source_mask + target_ids
    ```

  - ### `load_model`

    ```python
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **init_kwargs)
    patch_model(model, tokenizer, model_args, is_trainable)
    # patch_model 的解释
    """
    model.generate = MethodType(PreTrainedModel.generate, model) 如果 generate 方法不是从 GenerationMixin 继承的，这行代码将 PreTrainedModel 类的 generate 方法动态地绑定到 model 对象上，以确保模型具有正确的生成功能。
    
    _resize_embedding_layer(model, tokenizer) 如果需要调整词汇表大小，这个函数会被调用，以调整模型的嵌入层大小以匹配新的词汇表。
    """
    ```

    + ###### ` init_adapter`

      接着看下 `Lora` 微调代码细节：

      ```python
      from peft import LoraConfig
      lora_config = LoraConfig(
          task_type=TaskType.CAUSAL_LM,
          inference_mode=False,
          modules_to_save=finetuning_args.additional_target,
          use_dora=finetuning_args.use_dora,
          **peft_kwargs,
      )
      model = get_peft_model(model, lora_config)
      ```

       步进 `PeftModelForCausalLM(PeftModel)` ,     `PeftModel(PushToHubMixin, torch.nn.Module)`, ` LoraModel(BaseTuner)`

      转到 `BaseTuner` 的`__init__`方法，执行了`inject_adapter` 函数，该函数用于创建适配器层（adapter layers）并将其替换为目标模块（target modules）。调用 `peft.mapping.get_peft_model` 函数时，如果传入了非提示调优（non-prompt tuning）的适配器类，则在内部自动调用的方法。

      ```python
      key_list = [key for key, _ in model.named_modules()
      # key_list 是模型结构模块, 列举如下            
      """
      'transformer.wte', 'transformer.drop', 'transformer.rotary_emb', 'transformer.h', 'transformer.h.0', 'transformer.h.0.ln_1', 'transformer.h.0.attn', 'transformer.h.0.attn.c_attn', 'transformer.h.0.attn.c_proj', 'transformer.h.0.attn...tn_dropout', 'transformer.h.0.ln_2', 'transformer.h.0.mlp'
      """      
      ```

      如果`peft_config`配置参数target_modules 中设置为 'c_attn'，则将模型中以 ' c_attn' 后缀结尾的 block，进行替换

      ```python
      parent, target, target_name = _get_submodules(model, key)
      
      print(parent)
      QWenAttention(
        (c_attn): Linear(in_features=2048, out_features=6144, bias=True)
        (c_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (attn_dropout): Dropout(p=0.0, inplace=False)
      )
      
      print(target)
      Linear(in_features=2048, out_features=6144, bias=True)
      
      print(target_name)
      c_attn
      ```

      接下来参数的核心是产生新的`module` 和取代旧的 `module`

      ```python
      self._create_new_module(lora_config, adapter_name, target, **kwargs)
      self._replace_module(parent, target_name, new_module, target)
      ```

      新的`module` 产生：

      ```python
      #  peft.tuners.lora.layer.py
      lora.Linear(
        (base_layer): Linear(in_features=5120, out_features=15360, bias=True)
        (lora_dropout): ModuleDict(
          (default): Identity()
        )
        (lora_A): ModuleDict(
          (default): Linear(in_features=5120, out_features=16, bias=False)
        )
        (lora_B): ModuleDict(
          (default): Linear(in_features=16, out_features=15360, bias=False)
        )
        (lora_embedding_A): ParameterDict()
        (lora_embedding_B): ParameterDict()
      )
      ```
      
      具体实施细节可以参看`LoraLayer`类和 `self.update_layer` 方法。
      
       `new_module`  结构如下：
      
      ```python
      new_module：
      lora.Linear(
        (base_layer): Linear(in_features=5120, out_features=15360, bias=True)
        (lora_dropout): ModuleDict(
          (default): Identity()
        )
        (lora_A): ModuleDict(
          (default): Linear(in_features=5120, out_features=16, bias=False)
        )
        (lora_B): ModuleDict(
          (default): Linear(in_features=16, out_features=15360, bias=False)
        )
        (lora_embedding_A): ParameterDict()
        (lora_embedding_B): ParameterDict()
      )
      ```
      
      ```python
          def _replace_module(self, parent, child_name, new_module, child):
              setattr(parent, child_name, new_module)
              if hasattr(child, "base_layer"):
                  child = child.base_layer
      
              if not hasattr(new_module, "base_layer"):
                  new_module.weight = child.weight
                  if hasattr(child, "bias"):
                      new_module.bias = child.bias
      
              if getattr(child, "state", None) is not None:
                  if hasattr(new_module, "base_layer"):
                      new_module.base_layer.state = child.state
                  else:
                      new_module.state = child.state
                  new_module.to(child.weight.device)
      
              for name, module in new_module.named_modules():
                  if (self.prefix in name) or ("ranknum" in name):
                      weight = child.qweight if hasattr(child, "qweight") else child.weight
                      module.to(weight.device)
      ```
      
      这块代码显示`module`替换的过程，setattr(parent, child_name, new_module) 这行代码将新模块 `new_module` 赋值给父模块 `parent` 的属性 `child_name`，从而替换掉原来的子模块 `child`. 
      
      `child = child.base_layer` 如果 `child` 是一个包装器，这行代码将 `child` 设置为它的基础层（原始模块)
      
      `weight = child.qweight if hasattr(child, "qweight") else child.weight` 这行代码检查原始模块是否有 `qweight` 属性，这通常表示原始模块的权重已经被量化。
  
  + `DataCollatorForSeq2Seq` 
  
    ```python
    """迭代器,用于构建dataloader"""
    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
    label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    ```
  
  + `CustomSeq2SeqTrainer`:  Initialize  Trainer
  
    ```python
    trainer = CustomSeq2SeqTrainer(
                        model=model,
                        args=training_args,
                        finetuning_args=finetuning_args,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        callbacks=callbacks,
                        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
                        **split_dataset(dataset, data_args, training_args),
    )
    """
    split_dataset(dataset, data_args, training_args)
    {'train_dataset': Dataset({
        featur...ows: 43
    }), 'eval_dataset': Dataset({
        featur...ows: 11
    })}
    """   	
    ```

+ #### pt 核心代码块

  这部分详细介绍下预训练的代码，结构流程与上节相似，差异性如下。

  `get_dataset` ：不需要使用`template`模板，启用了“打包”（packing）功能，样本将被组织成一系列的组，每组内的文本项将是`cutoff_len` 长度，每个文本项之间用`eos_token`分隔.

  ```
   text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]
  ```

   提供三种训练方法： `full`  ,`lora`  ,`freeze`  三种方法，主要介绍后者，即只调模型中少量的`block`模块。

  + `freeze`  ：

    ```python
           freeze_modules = {"all"}
           for name, _ in model.named_modules():
                if ".0." in name:
                    freeze_modules.add(name.split(".0.")[-1].split(".")[0])
                elif ".1." in name:  # MoD starts from layer 1
                    freeze_modules.add(name.split(".1.")[-1].split(".")[0])     
    """
    name:模型中 block的命名.例如：
    transformer.h.0
    transformer.h.0.ln_1
    transformer.h.0.attn
    transformer.h.0.attn.c_attn
    transformer.h.0.attn.c_proj
    transformer.h.0.attn.attn_dropout
    transformer.h.0.ln_2
    transformer.h.0.mlp
    transformer.h.0.mlp.w1
    transformer.h.0.mlp.w2
    transformer.h.0.mlp.c_proj  
    """
    # 执行后 freeze_modules ：{'ln_1', 'all', 'mlp', 'ln_2', 'attn'}     
            trainable_layers = []
            for module_name in finetuning_args.name_module_trainable:
                if module_name not in freeze_modules:
                    raise ValueError(
                        "Module {} is not found, please choose from {}".format(module_name, ", ".join(freeze_modules))
                    )
    
                for idx in trainable_layer_ids:
                    trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))
    # 执行后确定可训练的模块 ['.22.', '.23.']
            for name, param in model.named_parameters():
                if any(trainable_layer in name for trainable_layer in trainable_layers):
                    if (not finetuning_args.pure_bf16) and (not finetuning_args.use_badam):
                        param.data = param.data.to(torch.float32)
                else:
                    param.requires_grad_(False)
    ```

    

+ #### 模型训练

  ```
  trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
  ```

  步入到 `_inner_training_loop` :

  step1:  加载数据迭代器     train_dataloader = self.get_train_dataloader()

  ```latex
  + 观察数据值 next(iter(train_dataloader))
  {'input_ids': tensor([[   854, 100817, 101923,  ..., 112277,  18493,  29490],
          [    23,   7948,     21,  ..., 100627, 104147, 100178]],
         device='cuda:0'), 
  'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
          [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'), 
  'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
          [1, 1, 1,  ..., 1, 1, 1]], device='cuda:0'), 
  'labels': tensor([[   854, 100817, 101923,  ..., 112277,  18493,  29490],
          [    23,   7948,     21,  ..., 100627, 104147, 100178]],device='cuda:0')}
  ```

  step2：设置训练控制变量 

  ```latex
   num_train_epochs
   num_update_steps_per_epoch
   max_steps
   num_examples
  ```

  假设`yaml`文件参数配置如下：

  ```yaml
  num_train_epochs    :   8 
  per_device_train_batch_size   : 2 
  gradient_accumulation_steps  :  2  
  eval_steps           :    10  
  save_steps           ：   500
  ```

  如果训练样本集合数量`num_examples = 43 `,  先计算整体`total_train_batch_size = per_device_train_batch_size *  gradient_accumulation_steps * n_gpus `。假设`gpu`数量设置1，那么总体`total_train_batch_size  = 4`, 迭代一轮内，扫描完全部的训练样本，参数更新所需要总步数：`num_update_steps_per_epoch = 43//4 = 11`

  `gradient_accumulation_steps` 作用是累积梯度，使得模型参数在一次更新之前能够基于多个批次的梯度进行更新。具体来说，当 `gradient_accumulation_steps` 设置为一个大于1的整数时，每个批次的梯度不会立即用来更新模型参数，而是累积起来。直到累积了 `gradient_accumulation_steps` 个批次的梯度后，这些梯度才会被用来计算参数的更新。这种方法在训练时可以减少内存的使用，因为每个批次的样本数量减少了，同时保持了较大的有效批量大小，这对于模型的收敛和性能是有益的。

  

  step3:   设置优化器和任务调度器  

  ```python
  self.create_optimizer_and_scheduler(num_training_steps=max_steps)
  ```

  step4:   状态，回调，控制

  + `TrainerState` : 这个类包含一个内部状态（inner state），该状态在模型和优化器进行检查点（checkpointing）保存时会被一同保存下来，并且会传递给 `TrainerCallback`，并传递给 [`TrainerCallback`]。

    ```python
    # 实例化
    self.state
    >>>
    TrainerState(epoch=0, global_step=0, max_steps=296, logging_steps=10, eval_steps=10, save_steps=500, train_batch_size=2, num_train_epochs=8, num_input_tokens_seen=0, total_flos=0, log_history=[], best_metric=None, best_model_checkpoint=None, is_local_process_zero=True, is_world_process_zero=True, is_hyper_param_search=False, trial_name=None, trial_params=None)
    ```

  + `TrainerCallback`： 这是一个内部类，它按顺序调用回调函数列表。

    ```latex
    # 实例化
    self.callback_handler
    # 可调用方法主要有：
    """
    30:'eval_dataloader'                31:'lr_scheduler'
    32:'model'                          33:'on_epoch_begin'
    34:'on_epoch_end'                   35:'on_evaluate'
    36:'on_init_end'					37:'on_log'
    38:'on_predict'						39:'on_prediction_step'
    40:'on_save'						41:'on_step_begin'
    42:'on_step_end'					43:'on_substep_end'
    44:'on_train_begin'					45:'on_train_end'
    46:'optimizer'						47:'pop_callback'
    48:'remove_callback'				49:'tokenizer'
    50:'train_dataloader'
    """
    ```

  + `TrainerControl`: 专门用来管理训练过程中控制流程的类。在训练过程中，可能需要根据某些条件改变流程，比如提前终止训练、改变学习率等。

    ```python
    # 实例化
    self.control
    >>>
    TrainerControl(should_training_stop=False, should_epoch_stop=False, should_save=False, should_evaluate=False, should_log=False)
    ```

       + ```
         # 控制开关更新顺序
         self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
         self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
         self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
         self.control = self.callback_handler.on_step_end(args, self.state, self.control)
         ...
         ...
         self.control = self.callback_handler.on_train_end(args, self.state, self.control)
         ```

进入训练循环，对每批数据进行Loss 求解，并更新梯度。

```python
with  self.accelerator.accumulate(model):
      tr_loss_step = self.training_step(model, inputs)
"""
tr_loss_step:
    tensor(1.4847, device='cuda:0')
"""  
```

整个训练代码逻辑已介绍完毕：

```python、
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

+ #### 模型评估

  ```mermaid
  flowchart TD
        _maybe_log_save_evaluate -.-> evaluate -.->
        evaluation_loop -.->  prediction_step
  ```

  

 





























.
