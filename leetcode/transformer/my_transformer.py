import torch
import torch.nn as nn
import torch.nn.functional as F

class MySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 这里用一个 Wqkv 简化实现，也可以分成三个 Linear
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq, d_model)
        """
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 变形为多头：(B, n_heads, T, head_dim)
        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, h, T, T)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, h, T, head_dim)

        # 合并 heads 回到 (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out)
        return out


class MyFFN(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MyTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = MySelfAttention(d_model, n_heads)
        self.ffn = MyFFN(d_model, dim_ff, activation="gelu")
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq, d_model)
        """
        if self.norm_first:
            # Pre-LN
            # 注意：真正 GPT/很多大模型都是 Pre-LN 架构
            attn_out = self.self_attn(self.norm1(x), attn_mask)
            x = x + self.dropout1(attn_out)

            ffn_out = self.ffn(self.norm2(x))
            x = x + self.dropout2(ffn_out)
        else:
            # Post-LN（论文原版）
            attn_out = self.self_attn(x, attn_mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))
        return x


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MyTransformerEncoderLayer(d_model, n_heads, dim_ff)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

def test():
    import torch
    import torch.nn as nn

    d_model = 50
    batch = 2
    seq_len = 10

    # 为每个 token 设置不同均值/方差
    means = torch.linspace(0, 30, steps=seq_len)       # 让 token 0 ~ token 9 的均值线性增加
    stds  = torch.linspace(1, 5, steps=seq_len)        # 让方差也逐渐增加

    # 构造形状为 (batch, seq_len, d_model) 的 x
    x = torch.zeros(batch, seq_len, d_model)
    for b in range(batch):
        for t in range(seq_len):
            x[b, t] = torch.randn(d_model) * stds[t] + means[t]

    print("原始向量均值（按 token）:")
    print(x.mean(dim=-1))  # shape: (batch, seq_len)

    print("\n原始向量方差（按 token）:")
    print(x.var(dim=-1, unbiased=False))

    # 定义 LayerNorm
    layer_norm = nn.LayerNorm(d_model)

    # 前向
    y = layer_norm(x)

    print("\nLN 后均值（应接近 0）:")
    print(y.mean(dim=-1))

    print("\nLN 后方差（应接近 1）:")
    print(y.var(dim=-1, unbiased=False))



def main():
    B, T, C = 2, 16, 64
    x = torch.randn(B, T, C)
    enc = MyTransformerEncoder(d_model=C, n_heads=8, dim_ff=256, num_layers=2)
    y = enc(x)
    print(y.shape)  # (2, 16, 64)

test()