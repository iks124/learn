# cbow_ns.py
# 运行: python cbow_ns.py
# 需要: pip install torch numpy

import os
import math
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#######################
# 配置参数（可调整）
#######################

from pathlib import Path
root = Path(__file__).resolve().parent.parent
INPUT_FILE = root / "data" /"zh.txt"      # 语料已用空格分词
OUTPUT_VECTORS = root / "code" /"zh_vectors.txt"
EMBED_SIZE = 200              # 词向量维度
WINDOW_SIZE = 4               # CBOW 上下文窗口半径
MIN_COUNT = 1                 # 低频阈值
NEGATIVE_SAMPLES = 5          # 每个正样本的负样本数
BATCH_SIZE = 512
EPOCHS = 10
LR = 0.025
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#######################
# 读取语料并建表
#######################
def read_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.split()   # 已分词
    return tokens

tokens = read_corpus(INPUT_FILE)
print(f"Total tokens read: {len(tokens)}")

counter = Counter(tokens)
vocab = {w: c for w, c in counter.items() if c >= MIN_COUNT}
sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
words = [w for w, c in sorted_vocab]
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)
print(f"Vocab size (freq>={MIN_COUNT}): {vocab_size}")

indexed_tokens = [word2idx[w] for w in tokens if w in word2idx]

#######################
# 生成 CBOW 训练样本
# 每个样本: (context_indices(list), target_index)
#######################
def generate_cbow_samples(indexed_tokens, window_size):
    samples = []
    n = len(indexed_tokens)
    for i in range(n):
        target = indexed_tokens[i]
        cur_w = random.randint(1, window_size)
        left = max(0, i - cur_w)
        right = min(n, i + cur_w + 1)
        ctx = []
        for j in range(left, right):
            if j == i:
                continue
            ctx.append(indexed_tokens[j])
        if len(ctx) == 0:
            continue
        samples.append((ctx, target))
    return samples

samples = generate_cbow_samples(indexed_tokens, WINDOW_SIZE)
print(f"Total training samples: {len(samples)}")

#######################
# 负采样分布（unigram^0.75）
#######################
counts = np.array([counter[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
pow_counts = counts ** 0.75
unigram_probs = pow_counts / pow_counts.sum()

#######################
# Dataset / DataLoader
#######################
class CBOWDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]
        return ctx, tgt

def collate_fn(batch):
    # batch: list of (ctx_list, tgt_idx)
    # padding ctx 到同长，并返回 mask
    max_len = max(len(ctx) for ctx, _ in batch)
    ctx_tensor = torch.full((len(batch), max_len), fill_value=-1, dtype=torch.long)  # -1 表示pad
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    tgt_tensor = torch.tensor([tgt for _, tgt in batch], dtype=torch.long)
    for i, (ctx, tgt) in enumerate(batch):
        L = len(ctx)
        ctx_tensor[i, :L] = torch.tensor(ctx, dtype=torch.long)
        mask[i, :L] = 1.0
    return ctx_tensor, mask, tgt_tensor

dataset = CBOWDataset(samples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                        num_workers=0, collate_fn=collate_fn)

#######################
# 模型：CBOW 输入嵌入（上下文） -> 平均 -> 与输出嵌入（目标）做NS
#######################
class CBOWNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_size)
        self.output_embeddings = nn.Embedding(vocab_size, embed_size)
        init_range = 0.5 / embed_size
        self.input_embeddings.weight.data.uniform_(-init_range, init_range)
        self.output_embeddings.weight.data.uniform_(0, 0)
    def context_embed(self, ctx_idx, mask):
        # ctx_idx: (B, L), mask: (B, L)
        # 将pad=-1的索引置零，不参与embedding；再用mask平均
        B, L = ctx_idx.shape
        # 替换 pad 索引为0，且 mask=0 保证不影响求和
        safe_ctx = ctx_idx.clone()
        safe_ctx[safe_ctx < 0] = 0
        emb = self.input_embeddings(safe_ctx)      # (B, L, D)
        emb = emb * mask.unsqueeze(-1)             # (B, L, D)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        ctx_mean = emb.sum(dim=1) / denom          # (B, D)
        return ctx_mean
    def target_out(self, tgt_idx):
        return self.output_embeddings(tgt_idx)

model = CBOWNS(vocab_size, EMBED_SIZE).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

#######################
# 训练循环
#######################
def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    for ctx_idx, mask, tgt_idx in dataloader:
        ctx_idx = ctx_idx.to(DEVICE)
        mask = mask.to(DEVICE)
        tgt_idx = tgt_idx.to(DEVICE)
        B = tgt_idx.size(0)

        # 负采样: (B, K)
        neg_samples = np.random.choice(vocab_size, size=(B, NEGATIVE_SAMPLES), p=unigram_probs)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()

        v_ctx = model.context_embed(ctx_idx, mask)          # (B, D)
        v_tgt = model.target_out(tgt_idx)                   # (B, D)
        v_neg = model.target_out(neg_samples)               # (B, K, D)

        # 正样本打分
        pos_score = torch.sum(v_ctx * v_tgt, dim=1)         # (B,)
        pos_log = F.logsigmoid(pos_score)

        # 负样本打分
        v_ctx_exp = v_ctx.unsqueeze(1)                      # (B,1,D)
        neg_score = torch.bmm(v_neg, v_ctx_exp.transpose(1,2)).squeeze(2)  # (B,K)
        neg_log = F.logsigmoid(-neg_score).sum(dim=1)       # (B,)

        loss = - (pos_log + neg_log).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    avg = total_loss / max(1, steps)
    print(f"Epoch {epoch}: avg loss = {avg:.6f}")
    return avg

print("Start training on device:", DEVICE)
for epoch in range(1, EPOCHS + 1):
    train_one_epoch(model, dataloader, optimizer, epoch)

#######################
# 保存词向量（输入端）
#######################
def save_vectors(path, model, idx2word):
    emb = model.input_embeddings.weight.data.cpu().numpy()
    vocab_size, dim = emb.shape
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{vocab_size} {dim}\n")
        for i in range(vocab_size):
            w = idx2word[i]
            vec = " ".join(f"{x:.6f}" for x in emb[i])
            f.write(f"{w} {vec}\n")
    print(f"Saved vectors to {path}")

save_vectors(OUTPUT_VECTORS, model, idx2word)
