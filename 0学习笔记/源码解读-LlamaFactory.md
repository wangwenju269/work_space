



#                                   llama-factory 官方代码

启动命令：将启动脚本保存为 `.sh` 文件`train.sh`，并在`dataset_info.json`里追加user数据路径，然后通过 `bash train.sh` 运行即可。

```shell
#!/bin/bash

deepspeed --include="localhost:2,3,4,5" src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --model_name_or_path ../model/qwen/Qwen2-7B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --dataset user \
    --template qwen \
    --cutoff_len 8192 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen2-7b/lora/sft \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500 \
    --ddp_find_unused_parameters false
```

下图仅关注代码中的函数和类，清晰地展示了它们的调用关系。

~~~mermaid
graph TD
    A[run_sft] --> B[load_tokenizer]
    A --> C[get_dataset]
    A --> D[load_model]
    A --> E[SFTDataCollatorWith4DAttentionMask]
    A --> H[CustomSeq2SeqTrainer]
    H --> J[train]
    H --> K[evaluate]
    H --> L[predict]
~~~



+ #### 准备项： 

  为了在调试时方便加载 `YAML` 配置文件中的参数，同时避免在命令行中手动输入大量参数，推荐以下方法：将 `YAML` 文件中的配置解析后动态加载到 `sys.argv` 中。

  ```python
  # 在train_bash.py 文件里顶部添加这两行代码
  import sys
  sys.argv.append("/workspace/LLaMA-Factory/config.yaml") # 这是 yaml 文件路径
  ```
  
  在 `LLaMA-Factory/src/llmtuner/hparams/parser.py` 文件中，定义了以下函数用于解析训练参数：
  
  ```python
  def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
      parser = HfArgumentParser(_TRAIN_ARGS)
      return _parse_args(parser, args)
  ```
  
  **`TRAIN_ARGS`**：  这是一个基于 `dataclass` 修饰的类，用于定义训练所需的参数配置。
  
  `_parse_args`: 用于解析命令行参数或配置文件（YAML 或 JSON），并将其转换为数据类实例。
  
  - 如果命令行参数是一个 YAML 文件（以 `.yaml` 结尾），调用 `parser.parse_yaml_file` 解析文件。
  - 如果命令行参数是一个 JSON 文件（以 `.json` 结尾），调用 `parser.parse_json_file` 解析文件。
  - 否则，调用 `parser.parse_args_into_dataclasses` 解析命令行参数。
  
  <details>
        <summary>
          <b>parse_args_into_dataclasses </b>
        </summary>
  
  `args` 参数是一个字符串列表，包含要解析的命令行参数，默认为 `sys.argv`。`return_remaining_strings` 为 `True` 时，返回未被解析的参数字符串列表。`look_for_args_file` 为 `True` 时，会在与脚本同名的目录下查找 `.args` 文件（如 `train_bash.args`）并加载其内容作为额外参数。`args_filename` 可指定自定义文件名替代 `.args`，`args_file_flag` 则从命令行中查找指定标志并加载其后的文件作为参数文件，以最后一次出现的文件为准。
  
  ```python
  for dtype in self.dataclass_types:
      keys = {f.name for f in dataclasses.fields(dtype) if f.init}  # 获取数据类的可初始化字段
      inputs = {k: v for k, v in vars(namespace).items() if k in keys}  # 提取匹配的参数
      for k in keys:
          delattr(namespace, k)  # 从 namespace 中移除已处理的参数
      outputs.append(dtype(**inputs))  # 实例化数据类并添加到输出列表
  
  # 如果必要参数缺失，argparse 会直接抛出错误，例如：
  # train_bash.py: error: the following arguments are required: --model_name_or_path, --output_dir                
  ```
  
  </details>
  
  ------
  
+ #### SFT 代码块

  - ##### load_tokenizer  

    ```python
    tokenizer = AutoTokenizer.from_pretrained(
                            model_args.model_name_or_path,
                            use_fast=model_args.use_fast_tokenizer,
                            split_special_tokens=model_args.split_special_tokens,
                            padding_side="right",
                            **init_kwargs,)	
    patch_tokenizer(tokenizer)
    ```

    `patch_tokenizer` 的作用是确保 `tokenizer` 对象具备正确的 `_pad` 方法实现。如果 `tokenizer` 未从 `PreTrainedTokenizerBase` 继承 `_pad` 方法，该方法会为其动态添加，以确保序列填充功能正常运作。

    ```python
    tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
    ```

  - ##### get_dataset

    **get_template_and_fix_tokenizer**:  首先根据模板名称获取对应的模板配置，处理停止词、EOS 和 PAD 令牌，然后动态生成 Jinja 模板以支持聊天格式的文本生成。

    **_get_merged_dataset**

    + *get_dataset_list*

      这函数代码通过解析 `dataset_info` 配置，动态构建 `DatasetAttr` 对象，设置数据集的格式、列名、标签等属性，并支持多种数据格式（如 `alpaca` 和 `sharegpt`）。它根据配置信息灵活处理通用属性、列名和标签，确保数据集属性完整且适配不同任务类型。

    + *_load_single_dataset*

      `load_dataset` 是 Hugging Face Transformers 库中用于加载数据集的核心函数。它支持从本地文件、Hugging Face Hub 或其他远程资源加载数据集，并提供了丰富的配置选项。以下是对该函数的详细解析及其底层实现逻辑。`load_dataset` 函数在幕后执行以下操作：

      1. **数据集脚本加载与缓存**：
         - 如果库中没有缓存，则从指定路径 (`path`) 下载并导入数据集脚本。
         - 如果数据集没有专用的数据集脚本，则会导入一个通用的数据集脚本（如 JSON、CSV、Parquet、文本等）。
         - 数据集脚本是小型 Python 脚本，定义了数据集的构建器。它们包含数据集的引用、信息和格式，以及原始数据文件的路径或 URL，并提供了加载数据示例的代码。
      2. **数据集脚本执行**：
         - 如果本地或缓存中尚不可用，从原始 URL 下载数据集文件。
         - 处理并将数据集缓存为类型化的 Arrow 表格。Arrow 表格是支持任意长度和嵌套对象的类型化表格，可以映射到 numpy、pandas 或 Python 通用类型。它们可以直接从磁盘访问，加载到 RAM 中，甚至可以通过网络流式传输。
      3. **返回请求的分割**：
         - 返回由 `split` 参数指定的数据集分割（默认返回所有分割）。

      ```
      dataset = load_dataset(
          path=data_path,          # 数据集路径，可以是本地目录或 Hugging Face Hub 上的数据集
          name=data_name,          # 数据集名称，通常用于指定 Hugging Face Hub 上的数据集版本
          data_dir=data_dir,       # 数据集所在的目录，指定后将在该目录中查找数据集
          data_files=data_files,   # 指定数据集文件的位置，可以是字典或列表
          split=data_args.split,   # 要加载的数据集分割，如 "train"、"validation"、"test" 等
          cache_dir=model_args.cache_dir,  # 缓存数据集的目录，用于快速加载
          token=model_args.hf_hub_token,   # Hugging Face Hub 的访问令牌，用于私有或认证数据集
          streaming=(data_args.streaming and (dataset_attr.load_from != "file")),  # 是否以流式方式加载数据集
          **kwargs                 # 其他关键字参数，用于处理特定情况或提供额外配置
      )
      ```

      **底层实现逻辑**：`load_dataset` 的具体实施细节较为复杂，主要分为以下两步：

      1. **模块加工** (`dataset_module_factory`)：
         - 根据传入的路径和参数，加载或生成数据集模块。
      2. **类构建** (`get_dataset_builder_class`)：
         - 获取数据集构建器类，用于后续的数据集加载和处理。

      ```
      Dataset({
          features: ['history', 'input', 'output', 'instruction'],
          num_rows: 54
      })
      ```

      返回的 `Dataset` 对象包含数据集的特征（`features`）和行数（`num_rows`），便于进一步的数据处理和分析。

    + *align_dataset*

      `align_dataset` 函数用于将输入的数据集（`Dataset` 或 `IterableDataset`）按照指定的格式对齐，并返回处理后的数据集。该函数的主要目的是将原始数据集转换为统一的格式，以便后续的训练或评估。

    **_get_preprocessed_dataset**
    
    这段代码的核心功能是对监督学习任务中的数据集进行预处理，将其转换为模型可以接受的输入格式。主要包括以下两个函数：
    
    1. **`preprocess_supervised_dataset`**：主函数，负责遍历数据集中的每个样本，调用 `_encode_supervised_example` 对样本进行编码，并构建模型输入。
    2. **`_encode_supervised_example`**：辅助函数，负责将单个样本（包括 `prompt`、`response`、`system` 和 `tools`）编码为模型所需的 `input_ids` 和 `labels`。
  
  - ##### load_model
  
    ```python
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **init_kwargs)
    ```
    
    + ###### ` init_adapter`
    
      以下是对 Lora 微调代码的详细解析，帮助理解其实现机制。
    
      1. **LoraConfig 配置**
      
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
      
    
      2. **模型结构遍历与适配器注入**
      
         在 `BaseTuner` 的 `__init__` 方法中，调用了 `inject_adapter` 函数，用于创建适配器层并替换目标模块。
      
         ```python
         key_list = [key for key, _ in model.named_modules()]
         ```
         
         <details> <summary><b>key_list包含了模型的所有模块路径</b></summary>
         
         ```python
              'transformer.wte', 'transformer.drop', 'transformer.rotary_emb', 'transformer.h', 'transformer.h.0', 'transformer.h.0.ln_1', 'transformer.h.0.attn', 'transformer.h.0.attn.c_attn', 'transformer.h.0.attn.c_proj', 'transformer.h.0.attn...tn_dropout', 'transformer.h.0.ln_2', 'transformer.h.0.mlp'
         ```
          </details>      
      
      3. **模块替换过程**

         ```python
          self._create_new_module(lora_config, adapter_name, target, **kwargs)
          self._replace_module(parent, target_name, new_module, target)
         ```
         <details> <summary> <b>新模块</b> </summary>
         
                ```
                  # peft.tuners.lora.layer.py
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
       </details>      


  + `DataCollatorForSeq2Seq` 

    一个名为 SFTDataCollatorWith4DAttentionMask 的数据整理器（data_collator），用于在训练或推理过程中对数据进行预处理和批处理. 
    ```python
    data_collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
    )
    ```
    
  + `CustomSeq2SeqTrainer`:  Initialize  Trainer
    
    这段代码初始化了一个名为 CustomSeq2SeqTrainer 的训练器（trainer），用于训练或微调一个序列到序列（Seq2Seq）模型。
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
    ```

------


+ #### pt 代码块

  这部分详细介绍下预训练的代码，结构流程与上节相似.
  `get_dataset`：不需要使用`template`模板，启用了“打包”（packing）功能，样本将被组织成一系列的组，每组内的文本项将是`cutoff_len` 长度，每个文本项之间用`eos_token`分隔.

  ```python
   text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]
  ```

  提供三种训练方法：`full`,`lora`,`freeze`三种方法，主要介绍后者，即只调模型中少量的`block`模块。

  + `freeze`  ：
    ```python
           freeze_modules = {"all"}
           for name, _ in model.named_modules():
                if ".0." in name:
                    freeze_modules.add(name.split(".0.")[-1].split(".")[0])
                elif ".1." in name:  # MoD starts from layer 1
                    freeze_modules.add(name.split(".1.")[-1].split(".")[0])     
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
	```python
          trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
	```
	直接步入到 `_inner_training_loop` 
	**step1:  加载数据迭代器  self.get_train_dataloader()**
  
    <details> <summary> <b>示例</b> </summary>
  
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
  </details>   
  
  **step2：设置训练控制变量** 
  
  <details> <summary> <b>示例</b> </summary>
  
    ```latex
    num_train_epochs
    num_update_steps_per_epoch
    max_steps
    num_examples
    ```
  </details>  
  
  `gradient_accumulation_steps` 作用是累积梯度，使得模型参数在一次更新之前能够基于多个批次的梯度进行更新。具体来说，当 `gradient_accumulation_steps` 设置为一个大于1的整数时，每个批次的梯度不会立即用来更新模型参数，而是累积起来。直到累积了 `gradient_accumulation_steps` 个批次的梯度后，这些梯度才会被用来计算参数的更新。这种方法在训练时可以减少内存的使用，因为每个批次的样本数量减少了，同时保持了较大的有效批量大小，这对于模型的收敛和性能是有益的。

  **step3: 设置优化器和任务调度器**  

  <details> <summary> <b>示例</b> </summary>
  
    ```python
    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    ```
  </details> 
  
  **step4:状态，回调，控制**

  `TrainerState`: 这个类包含一个内部状态（inner state），该状态在模型和优化器进行检查点（checkpointing）保存时会被一同保存下来，并且会传递给 `TrainerCallback`，并传递给 [`TrainerCallback`]。
  <details> <summary> <b>示例</b> </summary>
  
    ```python
    # 实例化
    self.state
    >>>
    TrainerState(epoch=0, global_step=0, max_steps=296, logging_steps=10, eval_steps=10, save_steps=500, train_batch_size=2, num_train_epochs=8, num_input_tokens_seen=0, total_flos=0, log_history=[], best_metric=None, best_model_checkpoint=None, is_local_process_zero=True, is_world_process_zero=True, is_hyper_param_search=False, trial_name=None, trial_params=None)
    ```
  </details>  

  <details> <summary> <b>TrainerCallback</b> </summary>

  
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
  </details> 
  
  `TrainerControl` 专门用来管理训练过程中控制流程的类。在训练过程中，可能需要根据某些条件改变流程，比如提前终止训练、改变学习率等。

  <details> <summary> <b>`TrainerControl`</b> </summary>

  
      ```python
      # 实例化
      self.control:TrainerControl(should_training_stop=False, should_epoch_stop=False, should_save=False, should_evaluate=False, should_log=False)
      self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
      self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
      self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
      self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
      self.control = self.callback_handler.on_step_end(args, self.state, self.control)
      ...
      ...
      self.control = self.callback_handler.on_train_end(args, self.state, self.control)
      ```
  </details> 
  
  **Loss 求解**
  
  ```python
  with  self.accelerator.accumulate(model):
        tr_loss_step = self.training_step(model, inputs)
  ```

------