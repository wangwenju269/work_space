## <center>**代码角度看QWen语言模型：模型细节与代码详解** </center>

![](./assets/qwen/QWenLMHeadModel.png)

 **本文将从模型细节（RMS Pre-Norm、SwiGLU激活函数、RoPE旋转位置编码），代码解读（model）以及推理等几个方面对千问模型的细节与代码详解，供大家一起参考，整个流程图上文所示。**

+  **`QWenLMHeadModel`** 

   `QWenLMHeadModel`类是调用模型的入口，继承关系：`QWenLMHeadModel` -->  `QWenPreTrainedModel` --> ` PreTrainedModel(tansformers)`

  先介绍初始化方法： 

  **`__init__  ` :**

     1.  初始化配置参数` config`,config是配置文件，从`transformer`加载传进来，提供推理模式，“bf 16” 'fp16' 'fp32'  可供选择 ，是否用 flash_attn 加速推理。举例：使用`bf16`精度

        ```python
        bf16 = torch.cuda.is_available() and  torch.cuda.is_bf16_supported()
        ```

     2.*核心部分*：首先构建32层自回归`transformer`解码器结构，其次将hidden 映射到所预测的 token，注意：这里的嵌入向量（ids->embedding 和 hidden->ids）不是 tie 在一起的，是两个单独独立的矩阵

  ```
  self.transformer = QWenModel(config)
  self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
  ```

  **`forward`**：

  ```
  # 重要核心代码块说明
         transformer_outputs = self.transformer(
              input_ids,                         # 输入ids                  
              past_key_values=past_key_values,   # cache 每层<key,values>,auto-regresive方式解码避免重复计算
              attention_mask=attention_mask,     # mask 矩阵，设置掩码策略，实际上是QWenAttention里的self.bais 起作用。
              token_type_ids=token_type_ids,     # 上下句的 id
              position_ids=position_ids,         # 绝对位置编码id，RoPE以绝对位置编码实现相对位置编码。
              head_mask=head_mask,               # 设置 None  
              inputs_embeds=inputs_embeds,
              encoder_hidden_states=encoder_hidden_states,
              encoder_attention_mask=encoder_attention_mask,
              use_cache=use_cache,
              output_attentions=output_attentions,
              output_hidden_states=output_hidden_states,
              return_dict=return_dict,
          )
          hidden_states = transformer_outputs[0]   # 获取最后层向量表示 
          lm_logits = self.lm_head(hidden_states)  # hiddden -> ids_logits
          loss = None
          if labels is not None:
              labels = labels.to(lm_logits.device)
              shift_logits = lm_logits[..., :-1, :].contiguous()         # 移位处理，求loss
              shift_labels = labels[..., 1:].contiguous()
              loss_fct = CrossEntropyLoss()  
              loss = loss_fct(
                  shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
              )
           ...
           ...
  ```

  **`get_output_embeddings` `set_output_embeddings` :**  这两个函数读取或设置头向量，估计是为了prefix-tuning 和P-tuning

  **`chat_stream`  **||   **`chat`**  ：  最终汇聚到**`generate`** ，而 `generate` 执行的 super().generate 函数，执行 sample 函数。

  ```
  class QWenLMHeadModel(QWenPreTrainedModel)
  class QWenPreTrainedModel(PreTrainedModel)
  class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin)
  # super().generate 实际上是执行 GenerationMixin类里的 generate 方法。
  ```

     自定义设置生成模式，默认是：`is_sample_gen_mode`  

  ```python
          is_sample_gen_mode = (
              (generation_config.num_beams == 1)
              and (generation_config.num_beam_groups == 1)
              and generation_config.do_sample is True
              and not is_constraint_gen_mode
              and not is_contrastive_search_gen_mode
          )    
  ```

    接着执行该类下 `sample`函数，get next token， 这里 self 就是 `QWenLMHeadModel` 模块

  ```python
              outputs = self(
                  **model_inputs,
                  return_dict=True,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
              )
  ```

   接着求 logits 和 概率分布

  ```
              probs = nn.functional.softmax(next_token_scores, dim=-1)
              next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
  ```

   

  | 生成文本的模式                 | 备注说明                    |
  | ------------------------------ | --------------------------- |
  | is_constraint_gen_mode         | 受限生成模式                |
  | is_contrastive_search_gen_mode | contrastive_search          |
  | is_greedy_gen_mode             | 贪心搜索模式                |
  | is_sample_gen_mode             | **torch.multinomial()采样** |
  | is_beam_gen_mode               | beam_search                 |
  | is_beam_sample_gen_mode        | beam_sample                 |
  | is_group_beam_gen_mode         | group_beam_search           |
  |                                |                             |

    官方代码

   

  ```
  is_constraint_gen_mode = (
      generation_config.constraints is not None or generation_config.force_words_ids is not None
  )
  is_contrastive_search_gen_mode = (
      generation_config.top_k is not None #top_k表示选取概率最高的k个token作为候选词
      and generation_config.top_k > 1
      and generation_config.do_sample is False #do_sample=True，则表示在生成文本时，从模型输出的所有词中随机采样一个作为下一个token
      and generation_config.penalty_alpha is not None
      and generation_config.penalty_alpha > 0 #penalty_alpha用于在contrastive search decoding中平衡model confidence and degeneration penalty
  )
  is_greedy_gen_mode = (
      (generation_config.num_beams == 1) #num_beams指定候选解的数量。Beam Search生成文本时，预测模型会对每一个时间步，生成一定数量的候选解
      and (generation_config.num_beam_groups == 1)#num_beam_groups代表在生成下一个单词时要使用的束搜索组数
      and generation_config.do_sample is False #do_sample=True，则表示在生成文本时，从模型输出的所有词中随机采样一个作为下一个token
      and not is_constraint_gen_mode #is_constraint_gen_mode=True表示对生成文本的内容进行现在
      and not is_contrastive_search_gen_mode
  )
  is_sample_gen_mode = ( 
      (generation_config.num_beams == 1)#num_beams指定候选解的数量。Beam Search生成文本时，预测模型会对每一个时间步，生成一定数量的候选解
      and (generation_config.num_beam_groups == 1)#num_beam_groups代表在生成下一个单词时要使用的束搜索组数
      and generation_config.do_sample is True#do_sample=True，则表示在生成文本时，从模型输出的所有词中随机采样一个作为下一个token
      and not is_constraint_gen_mode
      and not is_contrastive_search_gen_mode
  )
  ...  
  ```

  

+ **`QWenModel`** 

   继承关系：`QWenModel` -->  `QWenPreTrainedModel` --> ` PreTrainedModel(tansformers)`

​		**`__init__  ` :**

​         +  初始化配置参数` config`：（vocab_size，num_hidden_layers，embed_dim，... ）

​         +  核心模块：

```
    self.wte = nn.Embedding(self.vocab_size, self.embed_dim)  # 加载 embedding 向量
    self.drop = nn.Dropout(config.embd_pdrop)                 # 
    self.h = nn.ModuleList(                                   # 32轮 block  
                [
                    QWenBlock(
                        config,
                        layer_idx=i,
                    )
                    for i in range(config.num_hidden_layers)
                ]
            )
      self.ln_f = RMSNorm(                                    # RMSNorm 操作
                self.embed_dim,
                eps=config.layer_norm_epsilon,
            )

```

**`forward`**：

```python
        inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)
        # self.h： block层数， past_key_values：每层 <key,values> 对
         for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
               ...
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                ) 
```

self.h 是整个`transformers`结构图，如下图所示。

![image-20231122184948035](./assets/qwen/image-20231122184948035.png)





+ **`QWenBlock`** 

 **`__init__  ` :**

```
 # 每一层block模块核心部分： RMSNorm -> Attention(RoPE) ->  residual -> RMSNorm -> ffn -> residual  牢记这是qwen使用tansformer的架构。
 # 不仅用 pre_Norm 也用了 Post_Norm 方法
 self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
 self.attn = QWenAttention(config, layer_number=layer_idx)     
 self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

 self.mlp = QWenMLP(config)
```

​    **forward :**

```python
        layernorm_output = self.ln_1(hidden_states)   #  RMSNorm 
        attn_outputs = self.attn(                     #  Attention(RoPE)
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = attn_output + residual        #  residual 
        layernorm_output = self.ln_2(layernorm_input)   #  RMSNorm 
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)         #  ffn
        hidden_states = residual + mlp_output           #  residual
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
```

+ **`RMSNorm`：**

  ​    RMS Norm相比于 layer Norm ，主要区别在于去掉均值的部分，计算公式如下。

  ![image-20231122115138652](./assets/qwen/image-20231122115138652.png)

​                            

```
                    class RMSNorm(torch.nn.Module):
                        def __init__(self, dim: int, eps: float = 1e-6):
                            super().__init__()
                            self.eps = eps
                            self.weight = nn.Parameter(torch.ones(dim))
                        # RMSNorm 核心计算公式 
                        # x :(batch, seq_len, hidden) 在hidden的维度，在 词向量 的维度做 Norm
                        def _norm(self, x):
                            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  

                        def forward(self, x):
                            if rms_norm is not None and x.is_cuda:
                                return rms_norm(x, self.weight, self.eps)
                            else:
                                output = self._norm(x.float()).type_as(x)
                                return output * self.weight
```





+  **`QWenAttention`** 

   **`__init__  ` :**

```python
# 自回归的方式，构建 下三角 attention 矩阵，即 mask 策略
# register_buffer 注册的变量 参数不会更新，会随着模型结构返回 和 nn.Parameter 有着明显的区别。
self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
# masked_bias       
self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
# self.projection_size是投影矩阵维度。一次性投影，获取query，key，value，三个矩阵表示。
self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
```

​    **forward :**

```python
 mixed_x_layer = self.c_attn(hidden_states)
 query, key, value = mixed_x_layer.split(self.split_size, dim=2)
 query = self._split_heads(query, self.num_heads, self.head_dim)
 key = self._split_heads(key, self.num_heads, self.head_dim)
 value = self._split_heads(value, self.num_heads, self.head_dim)
 # 以上是“切头” 操作，不难理解。
 # “切头”后，矩阵维度说明： query.shape == torch.Size([bs,seq_len,head_num,dim])
 # 注意：获取旋转位置矩阵， 代码中给出 ntk_alpha 的计算方式，‘有待后续理解’。
 rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(hidden_states.device)  
 ...
 #(后续是常规的 attention 操作)
 def _attn(self,query,key,value,attention,head_mask = None):
     # query,key,value, 维度[bs, head_num, seq_len,dim]
     attn_weights = torch.matmul(query,key.permute(0,1,3,2))
     if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
                       [],
                       value.size(-1) ** 0.5,
                       dtype = attn_weights.dtype,
                       device = attn_weights.device)
    query_length , key_length = query.size(-2), key.size(-2)
    # mask 矩阵
    causal_mask = self.bias[::,key_length - query_length: key_length, :key_length]
    mask_values = torch.finfo(attn_weights.dtype).min
    mask_values = torch.full([],mask_values,dtype = attn_weights.dtype).to(attn_attn_weights.devices)
    attn_weights = torch.where(causal_mask,attn_weights.to(attn_weights.dtype), mask_value)
    attn_weights = nn.functional.softmax(attn_weights,dim = -1).type(values.type)
    attn_weights = self.dropout(attn_weights)
    attn_output = torch.matmul(attn_weights,values)
    attn_output = attn_output.transpose(1,2)
    return attn_output
```

**`RotaryEmbedding`**

这块需要从原理上去理解，详细可见苏神的 `reformer`, 强烈推荐：[苏神](https://kexue.fm/archives/8265)

原理：`RoPE` 位置编码就是通过将key、query 向量逆时针旋转 m * theta 个角度，赋予位置 m点 的位置信息（通过绝对编码的方式实现相对编码）。 

采用 Dynamic NTK 等长度外推方法，本质上是改变 base，影响每个位置对应的旋转角度，进而影响模型的位置编码信息，最终达到长度外推的目的。

说明一下：base的不同取值会影响注意力远程衰减的程度。

+  太小的base也会破坏注意力远程衰减的性质，例如base=10或100时，注意力分数不再随着相对位置的增大呈现出震荡下降的趋势。base=1时，将完全失去远程衰减特性。
+ 当base较大时，随着base的提升，远程衰减的程度会逐渐削弱。base 越大，模型处理的输入长度越长。但是更大的base会造成注意力远程衰减的能力变弱，改变模型的注意力分布，（因为旋转矩阵是作用在<key,query>上 ），进而改变模型的输出质量。

**绝对位置实现相对位置的公式推导公式如下**	

![image-20231123120410784](./assets/qwen/image-20231123120410784.png)

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) 
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            # 动态的调整base值，需要后续理解三角函数内插和外插很数学的东西。
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )    
            # inv_freq 就是那个旋转的角度 theta, 
            # 采用原始transformer论文里的 base 值，原论文是加性运算。而 RoPE 是乘性运算。
            # 两者都能实现相对位置编码的功能。
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)   # 计算 m * theta 
            # Different from paper, but it uses a different permutation in order to obtain the same calculation  
            # 理解半天：原来是 dim 向量分量 中的 element 发生交换 
            emb = torch.cat((freqs, freqs), dim=-1) 
            from einops import rearrange
            self._rotary_pos_emb_cache = rearrange(emb, "n d -> 1 n 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len] 
```

+  **`QWenMLP`** 

  大模型常用的`SwiGLU`激活函数, 本质就是`swish`作为激活函数`glu`的变体，可以理解就是增加 ‘’门控‘’机制。代码很容易理解：

```python
class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.ffn_hidden_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.ffn_hidden_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.ffn_hidden_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output
```





## 分词

重点介绍 `AutoTokenizer` 类中 `from_pretrained` 的方法。

**step1 : let's see whether the tokenizer_type is passed so that we can leverage it**

 查看所使用的分词器是否在预设的变量 `TOKENIZER_MAPPING_NAMES` 里。

![](F:/git_work_project/work_project/知识学习笔记/assets/qwen/image-20231122184948035.png)

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

![image-20231127173818687](F:/git_work_project/work_project/知识学习笔记/assets/qwen/image5.png)

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







