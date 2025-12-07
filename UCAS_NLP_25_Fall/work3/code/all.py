import os
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    def __init__(self, data_dir: str, vocab: Dict[str, int] = None, tag2id: Dict[str, int] = None, split: str = "train"):
        if split == "train":
            corpus_path = os.path.join(data_dir, "train_corpus.txt")
            label_path = os.path.join(data_dir, "train_label.txt")
        elif split == "test":
            corpus_path = os.path.join(data_dir, "test_corpus.txt")
            label_path = os.path.join(data_dir, "test_label.txt")

        with open(corpus_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        assert len(texts) == len(labels), "文本和标签行数必须一致"

        self.sentences: List[List[str]] = []
        self.tags: List[List[str]] = []

        for t_line, l_line in zip(texts, labels):
            # 默认按空格分词；如果没有空格，则按字符切分
            t_items = t_line.split()
            if len(t_items) <= 1:
                t_items = list(t_line)
            l_items = l_line.split()

            assert len(t_items) == len(l_items), f"文本和标签长度不一致: {t_line} || {l_line}"
            self.sentences.append(t_items)
            self.tags.append(l_items)

        # 构建词表
        if vocab is None:
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            for sent in self.sentences:
                for ch in sent:
                    if ch not in self.vocab:
                        self.vocab[ch] = len(self.vocab)
        else:
            self.vocab = vocab

        # 构建标签映射
        if tag2id is None:
            tag_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
            self.tag2id = {tag: i for i, tag in enumerate(tag_list)}
        else:
            self.tag2id = tag2id

        self.id2tag = {i: t for t, i in self.tag2id.items()}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chars = self.sentences[idx]
        tags = self.tags[idx]
        x = torch.tensor([self.vocab.get(ch, self.vocab["<UNK>"]) for ch in chars], dtype=torch.long)
        y = torch.tensor([self.tag2id[tag] for tag in tags], dtype=torch.long)
        return x, y


def pad_collate_fn(batch, pad_idx: int, tag_pad_idx: int = -1):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)

    padded_x = torch.full((len(xs), max_len), pad_idx, dtype=torch.long)
    padded_y = torch.full((len(xs), max_len), tag_pad_idx, dtype=torch.long)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        l = len(x)
        padded_x[i, :l] = x
        padded_y[i, :l] = y
        mask[i, :l] = 1

    return padded_x, padded_y, mask


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int,
                 embedding_dim: int = 100, hidden_dim: int = 256,
                 pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # 转移矩阵：transitions[i, j] = 从标签 i 转移到标签 j 的得分
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))

    def _compute_emissions(self, x):
        embeds = self.embedding(x)                     # (batch, seq_len, emb_dim)
        lstm_out, _ = self.lstm(embeds)               # (batch, seq_len, hidden_dim)
        emissions = self.hidden2tag(lstm_out)         # (batch, seq_len, tagset_size)
        return emissions

    @staticmethod
    def _log_sum_exp(tensor, dim):
        max_score, _ = tensor.max(dim)
        max_score_broadcast = max_score.unsqueeze(dim)
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score_broadcast), dim))

    def _compute_log_partition(self, emissions, mask):
        """
        前向算法计算 log Z(x)
        emissions: (batch, seq_len, tagset_size)
        mask: (batch, seq_len)
        """
        batch_size, seq_len, num_tags = emissions.size()
        log_alpha = emissions[:, 0]  # (batch, num_tags)

        for t in range(1, seq_len):
            emit_scores = emissions[:, t].unsqueeze(1)   # (batch, 1, num_tags)
            trans_scores = self.transitions.unsqueeze(0) # (1, num_tags, num_tags)
            score_t = log_alpha.unsqueeze(2) + trans_scores + emit_scores  # (batch, num_tags, num_tags)
            log_alpha_t = self._log_sum_exp(score_t, dim=1)                # (batch, num_tags)

            mask_t = mask[:, t].unsqueeze(1)  # (batch, 1)
            log_alpha = log_alpha_t * mask_t + log_alpha * (~mask_t)

        return self._log_sum_exp(log_alpha, dim=1)  # (batch,)

    def _compute_gold_score(self, emissions, tags, mask):
        """
        计算金路径 score
        emissions: (batch, seq_len, num_tags)
        tags: (batch, seq_len)
        mask: (batch, seq_len)
        """
        batch_size, seq_len, num_tags = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)

        # 发射得分
        for t in range(seq_len):
            emit_t = emissions[:, t, :]          # (batch, num_tags)
            tag_t = tags[:, t]                   # (batch,)
            mask_t = mask[:, t].float()
            score += emit_t.gather(1, tag_t.unsqueeze(1)).squeeze(1) * mask_t

        # 转移得分
        for t in range(1, seq_len):
            prev_tag = tags[:, t - 1]
            curr_tag = tags[:, t]
            mask_t = (mask[:, t] & mask[:, t - 1]).float()
            trans_score = self.transitions[prev_tag, curr_tag]
            score += trans_score * mask_t

        return score

    def neg_log_likelihood(self, x, tags, mask):
        emissions = self._compute_emissions(x)
        log_Z = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        return (log_Z - gold_score).mean()

    def forward(self, x, mask):
        """
        解码（Viterbi），返回最优标签序列（list of list[int]）
        x: (batch, seq_len)
        mask: (batch, seq_len)
        """
        emissions = self._compute_emissions(x)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        viterbi_score = emissions[:, 0]  # (batch, num_tags)
        viterbi_path = []

        for t in range(1, seq_len):
            broadcast_score = viterbi_score.unsqueeze(2)      # (batch, num_tags, 1)
            broadcast_trans = self.transitions.unsqueeze(0)   # (1, num_tags, num_tags)
            score_t = broadcast_score + broadcast_trans       # (batch, num_tags, num_tags)
            best_score, best_path = score_t.max(1)            # (batch, num_tags), (batch, num_tags)
            viterbi_score = best_score + emissions[:, t]      # (batch, num_tags)
            viterbi_path.append(best_path)

        best_tags_list = []
        for i in range(batch_size):
            seq_len_i = int(mask[i].sum().item())
            last_score, last_tag = viterbi_score[i].max(0)
            best_tags = [last_tag.item()]
            for back_t in reversed(viterbi_path[: seq_len_i - 1]):
                last_tag = back_t[i][best_tags[-1]]
                best_tags.append(last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list


# ===== 评价函数：precision / recall / F1 =====

def compute_prf_per_class(true_ids, pred_ids, num_labels: int):
    """
    true_ids, pred_ids: List[int]，已展开的所有 token 标签（不含 padding）
    返回：
        per_class: {label_id: (P, R, F1, support)}
        micro: (P, R, F1)
    """
    assert len(true_ids) == len(pred_ids)
    import math

    tp = [0] * num_labels
    fp = [0] * num_labels
    fn = [0] * num_labels
    support = [0] * num_labels

    for t, p in zip(true_ids, pred_ids):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    total_tp = total_fp = total_fn = 0

    for i in range(num_labels):
        if tp[i] == 0 and fp[i] == 0:
            prec = 0.0
        else:
            prec = tp[i] / (tp[i] + fp[i])

        if tp[i] == 0 and fn[i] == 0:
            rec = 0.0
        else:
            rec = tp[i] / (tp[i] + fn[i])

        if prec == 0.0 and rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        per_class[i] = (prec, rec, f1, support[i])

        total_tp += tp[i]
        total_fp += fp[i]
        total_fn += fn[i]

    # micro-average
    if total_tp == 0 and total_fp == 0:
        micro_p = 0.0
    else:
        micro_p = total_tp / (total_tp + total_fp)

    if total_tp == 0 and total_fn == 0:
        micro_r = 0.0
    else:
        micro_r = total_tp / (total_tp + total_fn)

    if micro_p == 0.0 and micro_r == 0.0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    return per_class, (micro_p, micro_r, micro_f1)


def evaluate(model, dataloader, device, id2tag: Dict[int, str], o_tag_id: int):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            # 用 CRF 解码
            best_paths = model(x, mask)  # list of list[int]
            batch_size, seq_len = x.size()
            for i in range(batch_size):
                seq_len_i = int(mask[i].sum().item())
                true_i = y[i, :seq_len_i].tolist()
                pred_i = best_paths[i][:seq_len_i]
                assert len(true_i) == len(pred_i)
                all_true.extend(true_i)
                all_pred.extend(pred_i)

    num_labels = len(id2tag)

    # ----- 1. 含 O 的所有类别 -----
    per_class_all, micro_all = compute_prf_per_class(all_true, all_pred, num_labels=num_labels)
    macro_p_all = sum(v[0] for v in per_class_all.values()) / num_labels
    macro_r_all = sum(v[1] for v in per_class_all.values()) / num_labels
    macro_f1_all = sum(v[2] for v in per_class_all.values()) / num_labels

    print("\n===== 含 O 类别的指标（所有标签） =====")
    print(f"Micro P/R/F1: {micro_all[0]:.4f} / {micro_all[1]:.4f} / {micro_all[2]:.4f}")
    print(f"Macro P/R/F1: {macro_p_all:.4f} / {macro_r_all:.4f} / {macro_f1_all:.4f}")
    for i, (p, r, f1, sup) in per_class_all.items():
        print(f"  Label {i} ({id2tag[i]}): P={p:.4f}, R={r:.4f}, F1={f1:.4f}, support={sup}")

    # ----- 2. 去掉 O 之后的指标 -----
    keep_ids = [i for i in range(num_labels) if i != o_tag_id]
    if not keep_ids:
        print("\n没有除 O 外的其它标签，无法计算去掉 O 的指标。")
        return

    # 重新累积只考虑非 O 类别的 TP/FP/FN（等价于在 compute_prf... 里只对这些类求和）
    # 直接过滤 true/pred 中的 O：只保留 true != O 或 pred != O ? ——更严格是：只考虑 true 属于实体标签的 token
    # 对于“去掉 O”的场景，一般是只对 true != O 的 token 做评估比较常见。
    filtered_true = []
    filtered_pred = []
    for t, p in zip(all_true, all_pred):
        if t != o_tag_id:  # 只统计真实是实体的 token
            filtered_true.append(t)
            filtered_pred.append(p)

    if len(filtered_true) == 0:
        print("\n无非 O 真值标签，无法计算去掉 O 的指标。")
        return

    per_class_wo, micro_wo = compute_prf_per_class(filtered_true, filtered_pred, num_labels=num_labels)
    # 只在 keep_ids 上做宏平均
    macro_p_wo = sum(per_class_wo[i][0] for i in keep_ids) / len(keep_ids)
    macro_r_wo = sum(per_class_wo[i][1] for i in keep_ids) / len(keep_ids)
    macro_f1_wo = sum(per_class_wo[i][2] for i in keep_ids) / len(keep_ids)

    print("\n===== 去掉 O 类别后的指标（只对实体标签） =====")
    print(f"Micro P/R/F1: {micro_wo[0]:.4f} / {micro_wo[1]:.4f} / {micro_wo[2]:.4f}")
    print(f"Macro P/R/F1: {macro_p_wo:.4f} / {macro_r_wo:.4f} / {macro_f1_wo:.4f}")
    for i in keep_ids:
        p, r, f1, sup = per_class_wo[i]
        print(f"  Label {i} ({id2tag[i]}): P={p:.4f}, R={r:.4f}, F1={f1:.4f}, support={sup}")


def train_model(data_dir: str,
                batch_size: int = 32,
                lr: float = 1e-3,
                num_epochs: int = 10,
                embedding_dim: int = 100,
                hidden_dim: int = 256,
                device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = NERDataset(data_dir)
    pad_idx = dataset.vocab["<PAD>"]
    tag_pad_idx = -1  # 标签的 padding

    def collate_fn(batch):
        return pad_collate_fn(batch, pad_idx=pad_idx, tag_pad_idx=tag_pad_idx)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn)

    # 为了评估时固定顺序，可以再建一个不 shuffle 的 dataloader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn)

    model = BiLSTM_CRF(
        vocab_size=len(dataset.vocab),
        tagset_size=len(dataset.tag2id),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, mask in dataloader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.clone()
            y[y == tag_pad_idx] = 0
            y = y.to(device)

            loss = model.neg_log_likelihood(x, y, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"\n========== Epoch {epoch}/{num_epochs}, loss = {avg_loss:.4f} ==========")

        # 每个 epoch 结束后做一次评估（在同一个训练集上）
        evaluate(model, eval_dataloader, device, dataset.id2tag, o_tag_id=dataset.tag2id["O"])

    # 训练结束后保存模型和词表
    save_path = os.path.join(data_dir, "bilstm_crf_ner.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": dataset.vocab,
        "tag2id": dataset.tag2id,
    }, save_path)
    print("\n模型已保存到", save_path)

def test_model(data_dir: str,
               batch_size: int = 32,
               device: str = None):
    # 在测试集上评估模型
    dataset = NERDataset(data_dir)
    pad_idx = dataset.vocab["<PAD>"]
    tag_pad_idx = -1  # 标签的 padding
    def collate_fn(batch):
        return pad_collate_fn(batch, pad_idx=pad_idx, tag_pad_idx=tag_pad_idx)
    test_dataset = NERDataset(data_dir, vocab=dataset.vocab, tag2id=dataset.tag2id, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # 加载模型
    model = BiLSTM_CRF(
        vocab_size=len(dataset.vocab),
        tagset_size=len(dataset.tag2id),
        embedding_dim=100,
        hidden_dim=256,
        pad_idx=pad_idx,
    ).to(device)
    checkpoint = torch.load(os.path.join(data_dir, "bilstm_crf_ner.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\n========== 在测试集上评估模型 ==========")
    evaluate(model, test_dataloader, device, dataset.id2tag, o_tag_id=dataset.tag2id["O"])

if __name__ == "__main__":
    data_dir = "./data"
    #train_model(data_dir)
    test_model(data_dir)
