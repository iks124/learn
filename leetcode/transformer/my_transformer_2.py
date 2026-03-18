import torch
import torch.nn as nn
import torch.nn.functional as F


# 错误示范
class mytransformerblock(nn.module):
    def __init__(self,dim):
        super.__init__()
        self.ln = torch.nn.LayerNorm(dim)
        self.q = torch.nn.Linear(dim,dim)
        self.k = torch.nn.Linear(dim,dim)
        self.v = torch.nn.Linear(dim,dim)
        self.ffn = torch.nn.Linear(dim,dim)

    def forward(self,x):
        # x.shape = (B,T,C)
        B,T,C = x.shape
        x = x + self.ln(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn_score = (q @ k.transpose(-1,-2) / C**0.5)
        attn_score = F.softmax(attn_score,-1)
        attn_out = attn_score @ v
        ffn_out = self.ffn(attn_out)
        return ffn_out + attn_out

        
