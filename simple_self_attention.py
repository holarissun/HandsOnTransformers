import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.o = nn.Linear(d_model,d_model)
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head
        self.sqrt_head_dim = self.head_dim ** 0.5
        self.n_head = n_head
        
    def forward(self, x, mask = None, causal_attn = True):
        # x is input of shape (batch_size, seq_length, d_model)
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2) # shape (B, HN, T, HD)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2) # shape (B, HN, T, HD)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2) # shape (B, HN, T, HD)
        
        attn_s = torch.matmul(q, k.transpose(-2, -1))/ self.sqrt_head_dim
        
        # adding masks
        if causal_attn == True:
            mask = torch.tril(torch.ones(T, T, device=x.device))
        if mask is not None:
            attn_s = attn_s.masked_fill(mask.unsqueeze(0).unsqueeze(0)==0, -float('inf'))
        attn_w = torch.softmax(attn_s, dim = -1) # shape (B, HN, T, T)
        
        attn_out = torch.matmul(attn_w, v) # B, HN, T, HD
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, C)
        
        return self.o(attn_out) # B, T, C
        
        