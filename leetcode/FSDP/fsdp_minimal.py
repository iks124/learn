import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def setup():
    dist.init_process_group("nccl")
    print(f"Rank {dist.get_rank()} initialized.")
    torch.cuda.set_device(dist.get_rank())

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        return self.net(x)

def main():
    
    setup()
    model = ToyModel().cuda()
    # 这一步是不是已经把模型参数完整load到对应 GPU 了？

    # 自动按参数量包层（防止每个 Linear 都单独通信）
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e6
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):
        x = torch.randn(8, 1024).cuda()
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()

        if dist.get_rank() == 0:
            print(f"step {step} done")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 fsdp_minimal.py
