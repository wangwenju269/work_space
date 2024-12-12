### **量化技术一网打尽**

+ #### **GPTQ**

  思想：**逐层量化，对层内所有参数逐个量化，每个参数量化后，需要适当调整层内其他未量化的参数，以弥补量化造成的精度损失。**

  首先先补充下`gptq` 的基础知识，附带详细数学推导。

  + ##### **基础1：`OBD`**

    原理：利用二阶导数信息度量模型参数的显著性，也就是度量删除模型某个参数对结果的影响，剪掉影响小的参数降低模型复杂度提高泛化能力。

    理论基石：损失函数泰勒级数展开
    $$
    L(w +\Delta w) = L(w) + g^T \Delta w + \frac{1}{2}\Delta w^T H\Delta w + O(||\Delta w||^3)
    $$
    参数变化带来的扰动可以表示为：
    $$
    \Delta E = L(w +\Delta w) - L(w) \\=
    g^T \Delta w + \frac{1}{2}\Delta w^T H\Delta w + O(||\Delta w||^3) \\
     = \sum_i g_i \Delta w_i + \frac{1}{2}\sum_ih_{ii}\Delta w^2_i + \frac{1}{2}\sum_{i\neq j}h_{ij}\Delta w_i \Delta w_j + O(||\Delta w||^3)
    $$
    等式中：
    $$
    g_i = \frac{\partial E}{\partial w_i} , h_{ij} = \frac{\partial ^2E}{\partial w_i \partial w_j}
    $$
    OBD 做了一些假设，对上式进行简化：

    - 假设目标函数是二阶的，所以我们不考虑高阶项 $ O(||\Delta w||^3)$
    - 假设模型训练已充分收敛，因此所有参数的一阶偏导均为 0： $𝑔_𝑖=0,∀𝑖$
    - 假设删除任意一个参数后，其他参数对目标函数的影响不变。也就是说，每个参数对目标函数的影响是独立的。不考虑交叉项： $ℎ_{𝑖𝑗}=0,∀𝑖,𝑗,𝑖≠𝑗$

    最终优化目标：
    $$
    \Delta E = \frac{1}{2}\sum_ih_{ii}\Delta w^2_i
    $$
    根据 $\Delta E$ 的变化，从小到大给参数排个序，这样就确定了参数剪枝的次序。

  + ##### **基础2：`OBS`**

    `OBS` 认为，参数之间的独立性不成立，要考虑交叉项. 思想就是寻找对模型影响最小的参数并置零，然后更新模型参数补偿置零参数带来的影响。
    $$
    \Delta E =  \frac{1}{2}\Delta w^T H\Delta w
    $$
    **删除一个权重** $𝑤_𝑞$ , 意味着$\Delta w$  第 q 维要变化$w_q$,则固定为 $−𝑤_𝑞$ , 引入一个约束条件：
    $$
    e_q^T.\Delta w + w_q = 0
    $$
     其中 $𝑒_𝑞$ 是一个 one-hot 向量，第 $q$ 个位置是 1 ，其余位置是 0。用 Lagrange 乘数法求解：
    $$
    L = \frac{1}{2}\Delta w^T H\Delta w + \lambda (e_q^T.\Delta w + w_q)
    $$
    $L$ 对 $\Delta w$ 求偏导，并等于0
    $$
    \partial L/\partial \Delta w = H \Delta w + \lambda e_q =0 \\
    \Delta w = -H^{-1} \lambda e_q
    $$
    引入$\Delta w$ 在位置q的值为 $\Delta w_𝑞$   , $[-H^{-1}]_{qq}$表示获取对角线位置$qq$的海森逆矩阵的值.
    $$
    \lambda = \frac{w_q}{[-H^{-1}]_{qq}}
    $$
    将$\lambda$ 带入可得：
    $$
    \Delta w = -\frac{w_q}{[-H^{-1}]_{qq}} H^{-1}_{,:q}
    $$
    带入求解$\Delta E$：
    $$
    \Delta E = \frac{1}{2} \frac{w_q^2}{[H^{-1}]_{qq}}
    $$
    然后就可以按照影响从小到大给参数排个序，这样就确定了参数剪枝的次序。同时，每次剪枝一个参数，其他的参数也按照 $𝛥𝑤$ 更新一次。

  + ##### **基础3：OBC**

    提出假设：参数矩阵的同一行参数互相之间是相关的，而不同行之间的参数互不相关。减少计算海森逆计算难度。

    

  + ##### **基础4：`OBQ`**

    将剪枝操作转换量化版本, 修改如下。
    $$
    \Delta w = -\frac{w_q-quant(w_q)}{[-H^{-1}]_{qq}} H^{-1}_{,:q} \\
    \Delta E = \frac{1}{2} \frac{(w_q-quant(w_q))^2}{[H^{-1}]_{qq}}
    $$
    基于此，`gptq`主要创新在于：

    +  OBS 采用贪心策略，先量化对目标影响最小的参数；但 GPTQ 发现直接按顺序做参数量化，对精度影响也不大。**参数矩阵每一行的量化可以做并行的矩阵计算**，时间复杂度由$O(d_{row}.d_{col}^3)$ 降低至 $O(max(d_{row}.d_{col}^2 , d_{col}^3))$

    + **Lazy Batch-Updates**，延迟一部分参数的更新，它能够缓解 bandwidth 的压力；

      即：将参数矩阵按每 128 列划分为一个个 group，量化某一列时，group 内的参数立即更新，而 group 后面的列只记录更新量，延迟更新。当一个  group 的参数全部量化完成，再统一对后面的所有参数做一次更新。这就是 Lazy Batch-Updates。

    + **Cholesky Reformulation**，用 Cholesky 分解求海森矩阵的逆，在增强数值稳定性的同时，不再需要对海森矩阵做更新计算。

    算法流程图如下：

    **Quantize $W$ given inverse Hessian $H^{−1} = (2XX^T + \lambda I)^{-1}$ and blocksize $B$.**

    $Q ← O_{d_{row}× d_{col}}$                                                                            // quantized output

    $E ← O_{d_{row} × B}$                                                                               // block quantization errors

    $H^{−1} ← Cholesky(H^{−1})^T $                                                        // Hessian inverse information

    $ for    \  i = 0, B, 2B, . . . do $

    ​        $for \ j = i, . . . , i + B − 1 \ do$

    ​               $Q_{:,j} ← quant(W_{:,j} )$                                                      // quantize column

    ​               $E_{:,j−i} ← (W_{:,j} − Q_{:,j} ) / [H^{−1}]_{jj}$                                  // quantization error

    ​              $W_{:,j:(i+B) }← W_{:,j:(i+B) }− E_{:,j−i} · H^{−1}_{j,j:(i+B)} $                // update weights in block

    ​        $end \ for$

    $W_{:,(i+B):} ← W_{:,(i+B):} − E · H^{−1}_{i:(i+B),(i+B):}$                                // update all remaining weights$

    $end \ for$

    `autogptq` 库主要代码实现

    ```python
    class GPTQ:
        # 求解 H 矩阵
        def add_batch(self,inp，out):
            """
            input: [batch,seq_len,input_hidden]
            out:[batch,seq_len,out_hidden]
            """
            ...pass ...
        	"""
        	将 batch，seq_len维度合并转置，转化后inp：[input_hidden,batch*seq_len]
        	最终海森矩阵维度 H: [input_hidden,input_hidden],即论文中[d_col * d_col]
        	"""
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())  
        def fasterquant(self,**kwargs):
             W = self.layer.weight.data.clone()  #量化权重值 copy
             ''''''
             # 利用 cholesky 稳定性求解 海森矩阵的逆
             H = torch.linalg.cholesky(H)
             H = torch.cholesky_inverse(H)
             H = torch.linalg.cholesky(H, upper=True) 
            
            # 采用 lazy_batch_updates 的方法，解决计算/通信比 效率低下问题，具体实现：
             for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1     # 每个blocksize 设置 128
     
                W1 = W[:, i1:i2].clone()       # W1 切片取 每个blocksize 待量化参数，即量化前参数
                Q1 = torch.zeros_like(W1)       # Q1 保存 量化后参数
                Err1 = torch.zeros_like(W1)     # 误差, 即 (w_q - quant(w_q)) / $[H^{-1}]_{qq}
                Losses1 = torch.zeros_like(W1)  # 损失,即 \Delta E
                Hinv1 = Hinv[i1:i2, i1:i2]      # 海森矩阵的逆
    
                for i in range(count):
                    w = W1[:, i]          # 待更新的 $w_q$
                    d = Hinv1[i, i]       # 取对角线元素值, $[H^{-1}]_{qq}$
    
                    if group_size != -1:
                        if not static_groups:
                            if (i1 + i) % group_size == 0:
                                # 获取量化的缩放因子，最大最小值,零点等值
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)
                               
                            if ((i1 + i) // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]
                    # 执行量化
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    # 对比公式，更方便理解
                    Losses1[:, i] = (w - q) ** 2 / d**2    
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                """
                一个 group 的参数全部量化完成,根据Err1 所记录更新值(规范化误差：(w - q) / d),对后续 block 的 weight 进行更新。 
                """
                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
                
            
    ```

+ #### AWQ:  Activation-aware Weight Quantization

  + 背景：先指出`gptq` 的缺点：它使用二阶信息来进行误差补偿，但它可能在重建过程中过拟合校准集，从而扭曲分布之外领域上的学习特征。
  + 核心思想：权重对于LLM的性能并不同等重要”的观察，存在约（0.1%-1%）显著权重对大模型性能影响太大，通过跳过这1%的显著权重（salient weight）不进行量化，可以大大减少量化误差。即：**通过保护更“重要”的权重不进行量化，从而在不进行训练的情况下提高准确率。**

  + 判断显著权重的方法：**根据激活分布而不是权重分布**。为了避免硬件效率低下的混合精度实现，作者推导出放大显著通道（salient channels）可以减少其相对量化误差的方法。

  作者核心原话：

  ```latex
  1. We observe that we can find 1% of the salient weights in LLMs by observing the activation distribution.
  2. Keeping the salient weights in FP16 can significantly improve the quantized performance (PPl from
            43.2 (left) to 13.0 (middle)), but the mixed-precision format is not hardware-efficient. 
  3. We follow the activation awareness principle and propose AWQ (right). 
  4. AWQ performs per-channel scaling to protect the salient weights.leading to reduced quantized error, 
  ```

  + **实验1：**通过保留1%的显著权重来改进LLM量化

    **结论：** 发现跳过具有较大 Norm（基于 $W$）的权重通道并不能显著提高量化性能，与随机选择效果类似仅少量改进。而根据激活幅度（magnitude）选     择权重可以显著提高性能，通过只保留 0.1%-1% 的较大激活所对应权重通道就能显著提高量化性能。

  + **实验2：**通过激活感知缩放保护显著权重。

    先看量化公式：N 是量化比特数， $\Delta$  是量化缩放比例。
    $$
    Y = Q(W)X \\
    Q(w) = \Delta · Round(\frac{W}{\Delta}) \\
    \Delta = \frac{max(|W|)}{2^{N-1}}
    $$
    假设 $w \in W$ , 现将$w$ 乘以$s$,  同时将$x$ 除以$s$, 计算
    $$
    Q(w·s) ·\frac{x}{s} = \Delta^{'}·Round(\frac{ws}{\Delta})·x·\frac{1}{s}                        		
    $$
    $ \Delta^{'}$   是应用 $s$ 后的新的量化缩放（scaler）因子。

    举例分析：

    ```python
     #假设权重w 满足标准正态分布
    W = [-0.5,0.25,0.56,0.78,0.94]
    N = 3
    delta = max([abs(w) for w in W]) 
    bit = 2**(N-1)
    delta = delta / bit
    ```

    ```python
    "执行结果"
    delta = 0.235
    ```

    ```python
    quantization = [round(w/delta) for w in W]
    de_quantization = [delta * q  for q in quantization] 
    ```

     

    ```python
    quantization = [-2, 1, 2, 3, 4]
    de_quantization = [-0.47, 0.235, 0.47, 0.705, 0.94]
    ```

    另 $s = 1.25 $  作用于 $W[2]$

    ```python
    
    W_s = [-0.5,0.25,0.56*1.25,0.78,0.94]
    delta_s = 0
    for w in W_s :
        val = abs(w) 
        if  val >= 1: val = 1   # 梯度裁剪   
        delta_s = max(delta_s, val) 
    delta_s =  delta_s / bit
    
    quantization_s = []
    for w in W_s:
        if abs(w) > 1 :
           quantization_s.append(round(1 / delta_s)) 
        else:   
           quantization_s.append(round(w / delta_s)) 
           
    de_quantization_s = [delta_s * q  for q in quantization_s]   
    de_quantization_s[2] = de_quantization_s[2] / s
    ```

    ```python
    delta_s  = 0.235
    quantization_s = [-2, 1, 3, 3, 4]
    de_quantization_s = [-0.47, 0.235, 0.564, 0.705, 0.94]
    ```

    **结论：**

    + 缩放单个元素 $w_{i} \in W$ 不会改变$W$ 的极值，即$\Delta^{'} \approx  \Delta $. 

    + $Round(·)$ (标记$RoundErr$)  不会发生改变，该误差服从均匀分布(0-0.5)，导致平均误差0.25 。

    + 显著权重相对误差较小
      $$
      E_{rr} = \Delta  · RoundErr \\
      E_{rr^{'}} = \Delta ^{'} · RoundErr · \frac{1}{s} \\ 
      \frac{E_{rr^{'}}}{E_{rr}}  = \frac{\Delta ^{'}}{ \Delta }·\frac{1}{s}
      $$

      ```python
      # 量化之前的
      W = [-0.5,0.25,0.56,0.78,0.94]
      # 量化后的
      de_quantization = [-0.47, 0.235, 0.47, 0.705, 0.94]
      # 当第2个元素乘以 s , 量化结果  
      de_quantization_s = [-0.47, 0.235, 0.564, 0.705, 0.94]
      """
      明显看到量化误差减少 
      0.56 -> 0.47
      0.56 -> 0.564
      """
      ```

      确定优化目标：
      $$
      s^{*} = arg \ min \ L(s) \\
      L(s) = ∥Q(W · s)(s^{−1} · X) − WX∥
      $$
      由于量化函数是不可微的，我们无法用反向传播直接优化问题。采用网格搜索的方法去优化：
      $$
      s = s_X^ \alpha ;  \alpha^{*} = arg \ min \ L(s_X^ \alpha)
      $$
      $\alpha$ 是超参数，取值【0，1】，平衡显著权重和非显著权重的程度。

      **总结优点：**

      ​        由于不用反向传播训练的方法，将很少依赖校准集，预防过拟合。需要更少的量化过程数据，并且可以保留在校准集分布之外的知识。

+ #### **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**		

  + **提出背景**：作者发现激活中存在一些离群值，它们的绝对值明显更大；并且这些离群值分布在少量的几个特征中，称为离群特征 (Emergent Features)。

    不论是 per-token（针对激活 x 而言：每行对应一个量化系数） 还是 per-channel （针对权重 w 而言：每列对应一个量化系数）量化，都会受到这些离群值的很大影响。既然只有少量的特征包含离群值，LLM.in8() 的思路是把这些特征拿出来单独计算，只对剩余特征做量化。

    + **实现过程：**

      流程如下：

      1. 从输入的隐含状态中，按列提取异常值 (离群特征，即大于某个阈值的值)。
      2. 对离群特征进行 FP16 矩阵运算，对非离群特征进行量化，做 INT8 矩阵运算；
      3. 反量化非离群值的矩阵乘结果，并与离群值矩阵乘结果相加，获得最终的 FP16 结果。

      ![](.\assets\PTQ\llm_int8.png)

      ```python
      import torch
      X = [[2,45,-1,-17,-1],[0,12,3,-63,2],[-1,37,-1,-83,0]]
      W = [[-1,0],[2,0],[0,-2],[3,-2],[-1,2]]
      X = torch.tensor(X)
      W = torch.tensor(W)
      # 计算原始结果 X * W
      out = torch.mm(X,W)
      print(out)
      """
      out:
      tensor([[  38,   34],
              [-167,  124],
              [-174,  168]])
      """
      ###########  执行量化操作  ############# 
      fp16_x = torch.tensor([[45,-17],[12,-63],[37,-83]])
      fp16_w = torch.tensor([[2,0],[3,-2]])
      sub8_x = torch.tensor([[2,-1,-1],[0,3,2],[-1,-1,0]])
      sub8_w = torch.tensor([[-1,0],[0,-2],[-1,2]])
      # 将 X,W 按元素的绝对值最大值, 并按照绝对值重新排序
      c_x, _ =  torch.max(torch.abs(sub8_x), dim = 0)
      c_w, _ =  torch.max(torch.abs(sub8_w), dim = 1)
      
      "fp16计算"
      fp16_out = torch.mm(fp16_x,fp16_w).to(dtype=torch.float32)
      """
      fp16_out:
      tensor([[  39.,   34.],
              [-165.,  126.],
              [-175.,  166.]])
      """
      c_q_x = (127 / c_x).reshape(1,-1)
      c_q_w = (127 / c_w).reshape(-1,1)
      int8_x = torch.round(torch.mul(sub8_x, c_q_x))
      int8_w = torch.round(torch.mul(sub8_w, c_q_w)) 
      int8_quantization = torch.mm(int8_x,int8_w) 
      # 反量化操作转化fp16
      out_product = torch.outer(c_x,c_w) 
      out_fp16 = torch.mm(out_product.type(torch.float32),int8_quantization)  / (127 * 127)
      dequan_out = fp16_out + out_fp16
      print(dequan_out)
      
      ```

      实验结果：

      ```
      # 原始输出：
      tensor([[  38,   34],
              [-167,  124],
              [-174,  168]])
      # 先量化，后反量化得到结果
      tensor([[  37.5079,   34.9763],
              [-167.2381,   127.4646],
              [-176.4921,   166.9764]])
      ```

      结论：量化前后几乎不变，量化损失基本很少，真TM牛逼。

      实践注意事项：$W$ $X$  这两个矩阵分块时，一定要注意分块后不能改变索引的次序。此外，$X$ 切出 $m$ 行，$W$ 要切出 $m$ 列。

      

   



​			



​					

