# HandsOnTransformers
attention is all you need!

This repo is created as a playground for practicing transformer implementations. Topics to cover:
1. - [x] self attention
2. - [x] multihead attention
3. - [x] cross attention
4. - [x] masked attention
5. - [x] multi query attention
6. - [ ] group query attention
7. multi head latent attention
8. - [x] encoder
9. - [x] decoder
10. decoder-only models
11. encoder-only models
12. encoder-decoder models
13. positional encoding
14. tokenization
15. normalization
16. sequence-prediction task
17. classification task
18. regression task
19. transformers for RL
20. DPO
21. MoE
22. other applications?
23. acceleration techniques
24. KV cache




### 1. SelfAttention Module

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.q = nn.Linear(d_model, d_model)
    self.k = nn.Linear(d_model, d_model)
    self.v = nn.Linear(d_model, d_model)
    self.o = nn.Linear(d_model, d_model)

  def forward(self, x):
    # if x.shape == seq_len, batch_size, d_model: x = x.permute(1,0,2)
    # x.shape: batch_size, seq_len, d_model
    q = self.q(x) # shape = batch_size, seq_len, d_model
    k = self.k(x) # shape = batch_size, seq_len, d_model
    v = self.v(x) # shape = batch_size, seq_len, d_model

    attn_w = F.softmax(torch.matmul(q, k.transpose(1,2)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)), dim = -1) # shape batch_size, seq_len, seq_len
    attn_out = torch.matmul(attn_w, v) # batch_size, seq_len, d_model

    out = self.o(attn_out)
    return out
```

### 2. MultiHeadAttention Module

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0
        self.head_dim = self.d_model // self.n_head
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q(x) # batch_size, seq_len, d_model
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)

        mha_w = F.softmax( torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), dim = -1) # batch_size, self.n_head, seq_len, seq_len

        attn_out = torch.matmul(mha_w, v) # batch_size, self.n_head, seq_len, self.head_dim
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        mha_out = self.o(attn_out)
        return mha_out

```

### 3. Cross-Attention

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0
        self.head_dim = self.d_model // self.n_head

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, query, key):
        batch_size_q, seq_len_q, _ = query.shape
        batch_size_k, seq_len_k, _ = key.shape
        assert batch_size_q == batch_size_k

        q = self.q(query) # batch_size, seq_len, d_model
        k = self.k(key)
        v = self.v(key)

        q = q.view(batch_size_q, seq_len_q, self.n_head, self.head_dim).transpose(1,2) # shape = batch_size_q, self.n_head, seq_len_q, self.head_dim
        k = k.view(batch_size_k, seq_len_k, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size_k, seq_len_k, self.n_head, self.head_dim).transpose(1,2)

        x_attn_w = F.softmax(
                torch.matmul(q, k.transpose(-2, -1)) /
                torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
                , dim = -1) # shape = batch_size, self.n_head, seq_len_q, seq_len_k
        x_attn_out = torch.matmul(x_attn_w, v) 
        x_attn_out = x_attn_out.transpose(1,2).contiguous().view(batch_size_q, seq_len_q, self.d_model)

        x_out = self.o(x_attn_out)
        return x_out
        
```

### 4. Adding Causal Masks

- bug here: in-place op will destroy the computational graph
- .detach(), .data, +=1 (and other in-place op) will destroy the computational graph.
- So, we should use out-of-place op like masked_fill rather than masked_fill_ when we need to track the gradient of attn_scores

```python3
def generate_causal_mask(self, attn_scores, seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool().unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    return attn_scores

attn_scores = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
attn_scores = self.generate_causal_mask(attn_scores, seq_len)
attn_w = F.softmax(attn_scores, dim=-1)


# more efficient implementation:
seq_len = attn_scores.shape[-1]
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()
attn_scores = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
attn_w = F.softmax(attn_scores, dim=-1)

```



### 5. Multi Query Attention

- Why multi-query attention?
- In inference time, MHA is not efficient and is computationally expensive (need multiple Key / Value heads)
- Solution of MQA: only having 1 kv_head, and n_head query

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, causal_transformers = True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.causal_transformers = causal_transformers
        assert self.d_model % self.n_head == 0
        self.head_dim = self.d_model // self.n_head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, self.head_dim)
        self.v = nn.Linear(d_model, self.head_dim)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, 1, self.head_dim).transpose(1,2).expand(-1,self.n_head, -1, -1)
        v = v.view(batch_size, seq_len, 1, self.head_dim).transpose(1,2).expand(-1,self.n_head, -1, -1)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if self.causal_transformers:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool().view(1,1,seq_len, seq_len)
            attn_score = attn_score.masked_fill(mask, float('-inf'))

        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn_w, v) # batch_size, n_head, seq_len, head_dim

        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.o(attn_out)
        return output

```



### 6. Grouped Query Attention 

#### very tricky step in the view / expand step between groups. Should be super careful!
- The shape of k starts with self.k(x).shape = batch_size, seq_len, n_kv_heads(n_group) * head_dim
- to reshape it and expand the dimension, there are two choices:
  1. view(batch_size, seq_len, n_kv_heads, 1, head_dim)
  2. view(batch_size, seq_len, 1, n_kv_heads, head_dim)
- 1 is correct, and 2 seems to be wrong. This is because if we do (1).expand(-1, -1, -1, q_per_kv, -1).view(batch_size, seq_len, n_heads, head_dim), each group get repeated (expanded) for q_per_kv times
- but if we use 2, each group will be repeated together, and the results seem to be randomized

- This is because torch.tensor([1,2]).view(2,1).expand(2,3).view(-1) -> 111 222 (element-wise repeat)
- but torch.tensor([1,2]).view(1,2).expand(3,2).view(-1) -> 12 12 12 (repeat as a group)

#### but actually, even with 2, we can still implement GQA --- there is no requirement in GQA implementation that groups have to be adjacent (query) heads. 

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, n_group):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = self.d_model // self.n_head
        assert self.d_model % self.n_head == 0
        self.n_kv_heads = n_group
        self.q_per_kv = self.n_head // self.n_kv_heads
        assert self.n_head % self.n_kv_heads == 0

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.v = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape

        q = self.q(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # shape = batch_size, n_head, seq_len, self.head_dim
        k = self.k(x).view(batch_size, seq_len, self.n_kv_heads, 1, self.head_dim).expand(-1, -1, -1, self.q_per_kv, -1).view(batch_size, seq_len, -1, self.head_dim).transpose(1,2) # shape = batch_size, n_head, seq_len, self.head_dim
        v = self.v(x).view(batch_size, seq_len, self.n_kv_heads, 1, self.head_dim).expand(-1, -1, -1, self.q_per_kv, -1).view(batch_size, seq_len, -1, self.head_dim).transpose(1,2) # shape = batch_size, n_head, seq_len, self.head_dim

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weights, v) # shape = batch_size, n_head, seq_len, head_dim
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)

        return self.o(out)

        

```





### 7. Transformer Encoder Layer
Key components of encoder layers:
- self-attention
- feed-forward network
- residual connection and layer-normalization

#### 7.1 No mask version... 

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

```

#### 7.2 masked version

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask = None):
        attn_output = self.self_attn(x, mask = src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0
        self.head_dim = self.d_model // self.n_head
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape
        q = self.q(x) # batch_size, seq_len, d_model
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)


        mha_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            mha_score = mha_score.masked_fill(mask, float('-inf')) # may need to use mask == 0 instead of mask for e.g., padding masks.
        mha_w = F.softmax(mha_score, dim = -1) # batch_size, self.n_head, seq_len, seq_len

        attn_out = torch.matmul(mha_w, v) # batch_size, self.n_head, seq_len, self.head_dim
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        mha_out = self.o(attn_out)
        return mha_out

```

### 8. Transformer Decoder Layer

```python3
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        attn_out = self.self_attn(x, mask = mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

```


