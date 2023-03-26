---
description: >-
  传统的RNN，GRU，LSTM他们都有一个问题，就是不能并行计算。同时虽然LSTM解决了长期依赖性的问题，但如果我们有一个很长很长的上万字的文本要处理，这个时候LSTM还是显得力不从心。Transformer模型很好的解决了上面的两个问题，它在2017年于论文Attention
  is All you Need [1]发表，之后用于Bert，GPT2，GPT3等模型中。
---

# Transformer

Transformer 是一种基于自注意力机制（self-attention mechanism）的深度神经网络，它是自然语言处理领域中的一项重要技术。最早由 Google 提出，已经被广泛应用于机器翻译、文本生成、语言模型等任务中。

Transformer 的核心思想是使用自注意力机制来实现对输入序列的编码和对输出序列的解码。自注意力机制可以让模型对输入序列中的不同位置进行关注，并将不同位置的信息整合起来。这种关注机制可以看作是一种在序列中进行“跨步”连接（skip connection）的方式，使得模型可以更好地捕捉序列中的长程依赖关系。



## Self-Attention 自注意力机制

Self-attention 的目的是根据输入序列中各个位置之间的依赖关系，计算出每个位置的特征向量表示，从而得到一个表示整个序列的矩阵表示（每个元素特征向量的拼接）。

Self-attention机制是一种将输入序列的不同部分关联起来的方法，可以在不引入循环或卷积结构的情况下学习到序列中的全局依赖关系。在self-attention中，每个输入元素都会计算一个注意力得分，该得分衡量它与其他元素之间的相对依赖性，并用这些得分来计算一个加权和。



### 序列自注意力计算的详细过程

在序列自注意力机制中，每个输入元素都可以被视为一个向量。对于每个向量，都可以通过一个矩阵变换来生成三个新向量：查询向量、键向量和值向量。

* 查询向量（query vector）：表示要计算相关度的向量，正如它的名字，它代表这个词作为查询时候的表示，每个词语都有一个查询向量；
* 键向量（key vector）：表示这个单词当作被比较的对象的表示向量，每个词语也有一个键向量；
* 值向量（value vector）：表示查询向量相关的向量，这里可以理解为一个更深层的表示，每个词语也有一个值向量

我们首先将查询向量与所有键向量进行点积运算，然后将结果除以一个可学习的缩放因子（为了使得梯度稳定），得到一组分数。这些分数可以视为查询向量与不同键向量之间的相似度分数，用于衡量它们之间的相关性。接下来，我们可以使用分数对值向量进行加权汇聚，以获得对查询向量的响应表示。

在序列自注意力机制中，每个输入元素都作为查询向量、键向量和值向量的来源，因此每个元素都可以被视为自身与序列中所有其他元素之间的关系的表示。通过这种方式，自注意力可以有效地捕捉序列中元素之间的长程依赖关系，从而在各种自然语言处理任务中取得了很好的效果。

假设我们有一个输入序列 $$x = {x_1, x_2, ..., x_n}$$，其中每个 $$x_i$$ 都是一个向量，维度为 $$d$$。我们可以通过一个线性变换来将每个向量映射到三个不同的向量，即查询向量 $$q_i$$、键向量 $$k_i$$ 和值向量 $$v_i$$：

$$
q_i = W_q x_i, \ k_i = W_k x_i, \ v_i = W_v x_i
$$

其中 $$W_q, W_k, W_v \in \mathbb{R}^{d \times d}$$ 是可学习的权重矩阵。

<figure><img src="../../.gitbook/assets/image (5) (1).png" alt=""><figcaption></figcaption></figure>



接下来，我们计算每对查询向量和键向量之间的点积得分 $$q_i  k_j，i=1,2,3...n,j=1,2,3...n$$，

<figure><img src="../../.gitbook/assets/image (12).png" alt=""><figcaption></figcaption></figure>

然后对值向量进行加权求和，以得到对查询向量的响应表示：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d}})V
$$

其中 $$Q = [q_1, q_2, ..., q_n] \in \mathbb{R}^{d \times n}$$ 是查询矩阵， $$K = [k_1, k_2, ..., k_n] \in \mathbb{R}^{d \times n}$$ 是键矩阵，$$V = [v_1, v_2, ..., v_n] \in \mathbb{R}^{d \times n}$$是值矩阵， $$\mathrm{softmax}$$  是对每行进行 softmax 操作， $$\sqrt{d}$$  是缩放因子，用于平衡点积得分的量级，使得梯度更加稳定。

<figure><img src="../../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

最终，对于输入序列中的每个元素 $$x_i$$，我们都可以通过自注意力机制它和序列中其他元素的注意力。

## 多头注意力

在序列模型里，输入序列被多次计算序列的自注意力，每次的计算都叫做一个头，每个头部可以关注序列中的不同部分，并计算出一个对每个位置的表示。然后，这些表示被合并成一个整体表示，以便于后续的模型处理。就像有多个人的大脑头部去关注不同的信息。

多头注意力是一种强大的注意力机制，可以帮助模型更好地理解和表示输入数据。

### 多头注意力的计算过程

在多头注意力中，我们需要多次计算序列的每个元素的自注意力。多头自注意力层中的每个头的 Q、K、V 矩阵都是由输入序列经过不同循环的线性变换得到的，每次循环的线性变换都不相同。每个头学习到不同的表示，从而可以捕捉序列中不同的特征。

例如，在进行机器翻译时，输入序列是源语言的词语序列，输出序列是目标语言的词语序列。在多头注意力机制中，可以将源语言的输入序列分别映射到不同的子空间中，如词性、词性和位置、句法结构等，每个子空间中的注意力头可以关注输入序列中与该子空间特征相关的信息，从而更好地捕捉源语言与目标语言之间的语义对应关系。



<figure><img src="../../.gitbook/assets/image (11) (1).png" alt=""><figcaption></figcaption></figure>

下面是多头注意力的计算过程：

将输入序列 $$X \in \mathbb{R}^{n \times d}$$通过 $$h$$ 个线性变换（称为“头”）转换为 $$h$$个查询 $$Q_1, Q_2, ..., Q_h$$、 $$h$$个键 $$K_1, K_2, ..., K_h$$ 和 $$h$$ 个值 $$V_1, V_2, ..., V_h$$，其中每个头的维度为 $$d/h$$。&#x20;

$$
Q_i = XW_i^Q，K_i = XW_i^K, V_i = XW_i^V,  i=1,2,...,h
$$

这里 $$W_i^Q \in \mathbb{R}^{d \times d/h}$$、 $$W_i^K \in \mathbb{R}^{d \times d/h}$$ 、 $$W_i^V \in \mathbb{R}^{d \times d/h}$$分别是用于将输入序列 $$X$$ 转换为查询 $$Q_i$$、键 $$K_i$$ 和值 $$V_i$$ 的线性变换矩阵， $$d$$ 是输入序列的维度， $$h$$ 是头的数量。每个头的维度为 $$d/h$$ ，因此每个头可以关注输入序列中的不同部分。

接下来，对于每个头 $$i$$，计算其注意力权重 $$Z_i$$，该权重表示该头在输入序列中关注的重要程度。这里采用前面说的点积注意力机制：&#x20;

$$
Z_i = \text{softmax}(\frac{Q_iK_i^T}{\sqrt{d/h}})
$$

$$\sqrt{d/h}$$ 是用于缩放点积的常数，旨在避免点积过大或过小而导致的梯度问题。

<figure><img src="../../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>

然后，将注意力权重 $$A_i$$与值 $$V_i$$ 相乘并相加，得到头 $$i$$ 的输出向量 $$O_i$$：&#x20;

$$
O_i = Z_iV_i
$$

最后，将所有头的输出向量拼接在一起，得到多头注意力的输出向量 $$O \in \mathbb{R}^{n \times d}$$：&#x20;

$$
Z = \text{Concat}(O_1, O_2, ..., O_h)
$$

多头注意力的输出向量 $$O$$ 可以作为下一层模型的输入，例如 Transformer 模型中的前馈神经网络。

<figure><img src="../../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>

多头注意力机制可以帮助模型更好地理解序列数据中的信息，从而提高模型的性能。

## Transformer

Transformer 包含两个主要模块：Encoder 和 Decoder。Encoder 模块将输入序列映射到一个高维空间中，而 Decoder 模块则根据 Encoder 模块生成的编码信息，逐步生成目标序列。

Encoder 级包含两个子层：多头自注意力层（Multi-Head Self-Attention Layer）和全连接前馈层（Fully Connected Feedforward Layer）。多头自注意力层用于对输入序列进行编码，全连接前馈层用于对编码后的序列进行进一步处理。Decoder 模块包含三个子层：多头自注意力层、编码器-解码器注意力层（Encoder-Decoder Attention Layer）和全连接前馈层。

Transformer 通过自注意力机制实现对序列的编码和解码，使得模型能够更好地捕捉序列中的依赖关系，进而提高自然语言处理等任务的效果。

<figure><img src="../../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

### Encoder

encoder主要用于将输入序列转换为一系列隐藏更深层的表示，这些隐藏深层表示可以用于进一步处理或生成输出序列。&#x20;

#### 多头自注意力层&#x20;

多头自注意力层是 Transformer 的核心部分，前面已经有详细的述说，它通过对输入序列中每个元素与所有元素的相似度进行计算，得到每个元素对于其他元素的权重，并使用这些权重进行加权平均，得到每个元素的向量表示。这个过程可以看做是将序列中的每个元素与其他元素进行“跨步”连接（skip connection），从而更好地捕捉序列中的长程依赖关系。

### 全连接前馈层&#x20;

全连接前馈层对应Transformer结构图中的Feed Forward，它是多头自注意力层的一个补充，用于对编码后的序列进行进一步处理，增强模型的表示能力。具体地，全连接前馈层包含两个线性变换，中间使用激活函数（如 ReLU）进行非线性变换，从而生成更加复杂的特征表示。

Transformer 的 Encoder 模块通过多头自注意力层和全连接前馈层对输入序列进行编码，从而捕捉序列中的依赖关系和特征表示。这些编码信息可以传递给 Decoder 模块，用于生成目标序列。

### Add\&Norm

Transformer中的Add & Norm是指在每个Multi-Head Attention和Feedforward层之后进行的一种规范化技术，目的是加快模型收敛速度并提高模型性能。这种想法来自于ResNet。

在Multi-Head Attention和Feedforward层中，模型进行一些线性变换和非线性变换，这些变换可能会导致梯度消失或梯度爆炸问题。为了解决这个问题，Transformer在每个层后添加了一个残差连接（residual connection），将输入和输出相加。在残差连接后，使用Layer Normalization对结果进行规范化。Layer Normalization是一种对数据进行归一化的方法，通过对每个特征维度上的数据进行标准化，使得不同特征维度上的数据具有相同的分布。最后，将归一化的结果与残差连接相加，得到该层的最终输出。

<figure><img src="../../.gitbook/assets/image (1) (3).png" alt=""><figcaption></figcaption></figure>

Add & Norm技术能够有效地减轻梯度消失和梯度爆炸问题，同时也有助于加速模型的收敛速度。

### Decoder

在Transformer模型中，outputs（shifted right）指的是模型输出序列中每个时间步的预测值，但是这些预测值与真实输出序列相比，都向右移动了一个时间步，因为这个序列需要先进入encoder。这种右移操作通常称为“右移一位”或“shifted right”。这种右移操作可以使得decoder的输入序列与encoder的输入序列相同。Decoder的作用是将编码器产生的高级特征的上下文向量（context vector）与目标序列中的单词一起，逐个地生成输出序列，将编码器的输出转化为人类可读的形式。

<figure><img src="../../.gitbook/assets/transformer_decoding_1.gif" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/transformer_decoding_2.gif" alt=""><figcaption></figcaption></figure>

### Masked Mult-Head Attention

Decoder的第一个子层是一个“masked”多头自注意力层，他的意思是掩盖，这意味着在计算注意力时，只允许当前位置之前的位置作为查询进行注意力计算，不允许当前位置之后的位置参与计算。这是因为在解码器中，我们需要逐步生成输出，而不是一次性生成所有输出。如果允许当前位置之后的位置参与计算，那么就相当于我们在生成当前位置的输出时使用了后面位置的信息，这会导致模型泄露未来信息，使得模型在生成输出时产生错误。 这种未来信息泄露会导致模型在生成输出时产生错误，因为模型会过度依赖未来信息，而忽略当前位置及之前的信息，从而导致模型对输入的理解出现偏差，输出的结果也就不准确了。 因此，为了避免这种情况，解码器中使用“masked”多头自注意力层来限制只使用当前位置及其之前的信息进行计算，保证每个位置的输出只受前面位置的影响，从而避免了未来信息的泄露。

<figure><img src="../../.gitbook/assets/image (1) (4).png" alt=""><figcaption></figcaption></figure>

### 位置编码

在Transformer模型中，为了将序列的位置信息引入模型中，需要对输入序列的每个位置进行编码。这是通过在输入序列中添加一个位置编码向量来实现的。位置编码向量可以被看作是一个与词向量同样维度的向量，其中每个元素的值是基于该位置以及每个维度的信息计算得到的。具体地，对于位置 $$pos$$和维度 $$i$$，位置编码向量 $$PE_{pos, i}$$ 的计算方式如下：

$$
PE_{\mathrm{pos,}i}=\begin{cases}\sin\left(\frac{px}{10000^{2/4}\mathrm{medel}}\right)&i\text{is even}\\ \cos\left(\frac{pg^{2/4}\mathrm{med}}{10000^{2(i-1)/d}\mathrm{med}}\right)&i\text{is odd}\end{cases}
$$

其中， $$PE_{pos, i}$$ 表示位置编码矩阵中位置 $$pos$$上的第 $$i$$ 维元素， $$d_{\text{model}}$$ 是词向量和位置编码向量的维度， $$pos$$  是当前位置的索引。公式中的 $$sin$$ 和 $$cos$$$ 函数分别代表正弦函数和余弦函数。它们能够给每个位置编码向量赋予一个独特的模式，从而区分不同位置的输入。在计算中，位置编码向量会被加到对应的词向量中，从而产生最终的输入向量。下面我们用一张图来直观感受一下这些模式。在下面的图中，每一行对应一个向量的位置编码。因此，第一行将是我们要添加到输入序列中第一个单词的嵌入中的向量。每行包含$$512$$个值 - 每个值的取值范围在$$1$$到$$-1$$之间。我们已经用彩色编码来使模式可见。

<figure><img src="../../.gitbook/assets/image (1) (2).png" alt=""><figcaption></figcaption></figure>

需要注意的是，由于位置编码向量是通过正弦和余弦函数进行计算的，所以在计算中不需要额外的训练，也不需要对每个位置编码向量进行更新。位置编码向量只需要在模型的初始化阶段计算一次，然后在每次输入序列的编码中使用即可。

下面我们用一个动画演示一下Transformer整个计算过程：

<figure><img src="../../.gitbook/assets/transform20fps.gif" alt=""><figcaption></figcaption></figure>

## Transformer的pytorch实现

首先，我们需要导入所需的库和模块：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

```

然后，我们定义Transformer模型的主要组件，包括编码器、解码器和整个Transformer模型本身。

在编码器和解码器中，我们实现了多头自注意力机制（multi-head self-attention）和前馈神经网络（feed-forward network）这两个核心组件。

在Transformer模型中，我们将编码器和解码器组合在一起，并添加一些额外的组件，如嵌入层（embedding layer）、位置编码器（position encoding）和输出层（output layer）。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 模型的维度
        self.n_heads = n_heads  # 多头注意力的头数
        self.d_k = d_model // n_heads  # 每个头的维度，保证能够整除
        
        # 创建权重矩阵
        self.W_Q = nn.Linear(d_model, d_model)  # 查询向量的权重矩阵
        self.W_K = nn.Linear(d_model, d_model)  # 键向量的权重矩阵
        self.W_V = nn.Linear(d_model, d_model)  # 值向量的权重矩阵
        
        # 最后的线性层
        self.W_O = nn.Linear(d_model, d_model)  # 输出向量的权重矩阵
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)  # 获取输入数据的批次大小
        
        # 通过线性层，分别计算 Q、K、V 的投影向量
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 将 Q、K、V 投影向量分裂为多个头
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # ...


```

在`MultiHeadAttention`类中，我们实现了多头自注意力机制，其中包含了以下主要部分：

1. `__init__` 方法：初始化函数，定义了模型的维度、多头注意力的头数、每个头的维度，并创建了权重矩阵（查询、键、值、输出）。
2. `forward` 方法：前向传播函数，用于计算多头自注意力机制的输出。在此函数中，我们首先通过线性层，将输入的 Q、K、V 分别投影到 d\_model 维度空间上。
3. 接着，我们将 Q、K、V 投影向量分裂为多个头，以便进行并行计算。
4. 然后，我们计算注意力得分 scores，通过将 Q 与 K 转置后相乘，再除以 math.sqrt(self.d\_k)。注意力得分用于计算每个值向量的权重，以便对值向量进行加权求和。

在这里，我们只是计算了注意力得分，并没有进行权重的计算。接下来，我们将使用 softmax 函数将注意力得分转换为权重:

```python
# 对 scores 进行缩放和掩码操作
if mask is not None:
mask = mask.unsqueeze(1)
scores = scores.masked_fill(mask == 0, -1e9)

# 将注意力得分进行 softmax 计算
    attn_weights = F.softmax(scores, dim=-1)
    
    # 将权重与 V 向量相乘
    attn_output = torch.matmul(attn_weights, V)
    
    # 将多头注意力向量拼接在一起
    attn_output = attn_output.view(batch_size, -1, self.d_model)
    
    # 通过最后的线性层，得到最终的多头注意力向量
    attn_output = self.W_O(attn_output)
    
    return attn_output, attn_weights
class FeedForward(nn.Module):
def init(self, d_model, d_ff):
super(FeedForward, self).init()
 # 创建两个线性层
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
def forward(self, x):
    # 通过 ReLU 激活函数
    x = F.relu(self.linear_1(x))
    x = self.linear_2(x)
    return x
class EncoderLayer(nn.Module):
def init(self, d_model, n_heads, d_ff, dropout=0.1):
super(EncoderLayer, self).init()
self.multihead_attn = MultiHeadAttention(d_model, n_heads)
self.ff = FeedForward(d_model, d_ff)
self.norm1 = nn.LayerNorm(d_model)
self.norm2 = nn.LayerNorm(d_model)
self.dropout1 = nn.Dropout(dropout)
self.dropout2 = nn.Dropout(dropout)
```

以上的代码是 Transformer 模型的一部分，包括了 MultiHeadAttention、FeedForward 和 EncoderLayer 三个类。下面是代码解释：

* MultiHeadAttention：多头注意力机制，将输入的 Q、K、V 矩阵分别通过线性变换得到 Q、K、V 的查询矩阵、键矩阵和值矩阵，然后计算注意力得分，并通过 softmax 函数将注意力得分转换为权重，最后将权重与 V 向量相乘得到多头注意力向量。
* FeedForward：前馈神经网络，通过两个线性层和 ReLU 激活函数对输入进行变换。
* EncoderLayer：编码器层，包括多头注意力机制、前馈神经网络、LayerNormalization 和 Dropout 层，其中 LayerNormalization 是为了减少训练过程中的内部协变量偏移，Dropout 是为了防止过拟合。

这三个类是 Transformer 模型的重要组成部分，可以用于语言建模、机器翻译等任务中。其中 MultiHeadAttention 的思想也被广泛应用于其他领域的深度学习模型中。





## Refernce 引用

1. Vaswani, Ashish et al. “Attention is All you Need.” _ArXiv_ abs/1706.03762 (2017): n. pag.
