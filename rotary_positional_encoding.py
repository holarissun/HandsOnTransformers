import torch
import torch.nn as nn

def sinusoidal_position_embedding(batch_size, n_head, max_len, d_model, device = 'cpu'):
    # position: (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (d_model//2) [0,1,2,...,255]
    ids = torch.arange(0, d_model // 2, dtype=torch.float)
    theta = torch.pow(10000, -2 * ids / d_model)

    # (max_len, d_model//2)
    embeddings = position * theta  # pos / (10000^(2i/d))
    # [32, 256]
    # it scales the embeddings by position (max value is not 1, but position in this case)
    
    # (max_len, d_model//2, 2)
    # [32, 256, 2]
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, d_model//2, 2) # e.g.,  [4, 8, 32, 256, 2]
    embeddings = embeddings.repeat((batch_size, n_head, *([1] * len(embeddings.shape))))

    # (bs, head, max_len, d_model)
    embeddings = torch.reshape(embeddings, (batch_size, n_head, max_len, d_model))
    embeddings = embeddings.to(device)
    return embeddings

# equivalent implementation of PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape max_len, 1
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        ) # shape: d_model/2, 

        pe[:, 0::2] = torch.sin(position * div_term) # auto broadcast, shape max_len, d_model/2
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register as buffer
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len, base = 10000):
        super().__init__()
        self.base = base
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.inv_freq = 1. / (self.base ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
    def forward(self, x):
        seq_len = x.shape[1]
        pe = torch.zeros((seq_len, self.embed_dim))
        freqs = torch.einsum('i,j -> ij', torch.arange(seq_len).float(), self.inv_freq)
        pe[:, 0::2] = torch.sin(freqs)
        pe[:, 1::2] = torch.cos(freqs)
        return x + pe.unsqueeze(0) # [bs, seq_len, embed_dim]
    

# RoPE implementation v1

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len, base = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.rotary_embeddings = torch.einsum("i,j->ij", torch.arange(self.max_seq_len).float(), self.theta)
        self.cos = torch.cos(self.rotary_embeddings) # [max_seq_len, dim//2]
        self.sin = torch.sin(self.rotary_embeddings) # [max_seq_len, dim//2]
        
    def forward(self, x):
        # query, key: [bs, seq_len, dim]
        # output: [bs, seq_len, dim]
        seq_len = x.shape[1]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim//2]
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim//2]
        
        def rotate_half(x):
            x = x.reshape(*x.shape[:-1], -1, 2)  # paired by 2: (batch, heads, seq_len, dim//2, 2)
            x1, x2 = x[..., 0], x[..., 1]
            return torch.stack([-x2, x1], dim=-1).reshape(*x.shape[:-2], -1)  # (batch, heads, seq_len, dim)
            # we start from [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
            
        # method 1
        # cos shape [1, 1, seq_len, dim//2]
        cos1 = cos.repeat_interleave(2, dim=-1) # cos1 shape [1, 1, seq_len, dim]
        sin1 = sin.repeat_interleave(2, dim=-1) # sin1 shape [1, 1, seq_len, dim]
        rope_x1 = x * cos1 + rotate_half(x) * sin1 # x shape batch, heads, seq_len, dim
    
        # method 2
        x = x.reshape(*x.shape[:-1], -1, 2) # batch, heads, seq_len, dim//2, 2
        rope_x2 = x * cos.unsqueeze(-1) + rotate_half(x) * sin.unsqueeze(-1) # reshaped cos: [1, 1, seq_len, dim//2, 1]
        
        rope_x = rope_x1 # or rope_x2
        return rope_x
    
# RoPE implementation v2
        
        
       
        


aa = torch.rand(256)
bb = torch.rand(32)

cc = torch.outer(aa, bb)
dd = torch.einsum('i,j->ij', aa, bb)
ee = aa.unsqueeze(1) * bb

assert torch.allclose(cc, dd)
assert torch.allclose(cc, ee)

