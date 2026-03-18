# minimal_fabric_fsdp.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---- 1) 一个会被 auto_wrap 的 Block ----
class Block(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab=32000, d_model=256, n_layers=4, block_size=128):
        super().__init__()
        self.vocab = vocab
        self.block_size = block_size
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model=d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)              # [B, T, C]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)               # [B, T, V]
        return logits

# ---- 2) 极简数据：随机 token ----
class RandomTokens(Dataset):
    def __init__(self, vocab=32000, block_size=128, n_samples=10_000):
        self.vocab, self.block_size, self.n_samples = vocab, block_size, n_samples

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.block_size + 1,), dtype=torch.long)
        return x

def main():
    devices = torch.cuda.device_count() or 1

    # ---- 3) FSDPStrategy：按 Block 自动 wrap，checkpoint 保存 full ----
    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full")

    fabric = L.Fabric(
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",
    )
    fabric.launch()
    fabric.seed_everything(3407)

    # ---- 4) dataloader ----
    vocab = 32000
    block_size = 128
    micro_batch = 8
    grad_acc_steps = 4
    dl = DataLoader(RandomTokens(vocab=vocab, block_size=block_size), batch_size=micro_batch, shuffle=False)
    dl = fabric.setup_dataloaders(dl)

    # ---- 5) model / optimizer ----
    with fabric.init_module(empty_init=False):
        model = TinyLM(vocab=vocab, block_size=block_size)

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=True)
    optimizer = fabric.setup_optimizers(optimizer)

    # ---- 6) train loop：grad accumulation + no_backward_sync ----
    model.train()
    step = 0
    iter_num = 0
    ckpt_path = "minimal_ckpt.pth"

    for batch in dl:
        input_ids = batch[:, :block_size].contiguous()
        targets   = batch[:, 1:block_size+1].contiguous()

        is_accumulating = (iter_num + 1) % grad_acc_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
            fabric.backward(loss / grad_acc_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if fabric.global_rank == 0:
                fabric.print(f"step={step} loss={loss.item():.4f}")

            # ---- 7) save checkpoint（演示）----
            if step == 5 and fabric.global_rank == 0:
                state = {"model": model, "optimizer": optimizer, "step": step, "iter_num": iter_num}
                fabric.save(ckpt_path, state)
                fabric.print(f"saved: {ckpt_path}")

        iter_num += 1
        if step >= 8:
            break

    # ---- 8) load checkpoint（演示）----
    fabric.barrier()
    if os.path.exists(ckpt_path):
        state = {"model": model, "optimizer": optimizer, "step": 0, "iter_num": 0}
        fabric.load(ckpt_path, state)
        if fabric.global_rank == 0:
            fabric.print(f"loaded ckpt: step={state['step']} iter={state['iter_num']}")

if __name__ == "__main__":
    main()
