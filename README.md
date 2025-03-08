# HandsOnTransformers
attention is all you need!


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

