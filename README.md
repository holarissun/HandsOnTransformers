# HandsOnTransformers
attention is all you need!

This repo is created as a playground for practicing transformer implementations. Topics to cover:
1. - [x] self attention
2. - [x] multihead attention
3. - [x] cross attention
4. - [x] masked attention
5. - [x] multi query attention
6. multi latent attention
7. decoder
8. encoder
9. positional encoding
10. tokenization
11. normalization
12. sequence-prediction task
13. classification task
14. regression task
15. transformers for RL
16. DPO
17. MoE
18. other applications?
19. acceleration techniques
20. KV cache




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
