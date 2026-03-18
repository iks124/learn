#!/usr/bin/env bash
set -e

# 导出环境变量（关键：添加 export，指定正确网卡）
#export TORCHDYNAMO_VERBOSE=1
#export TORCHDYNAMO_DISABLE=1
#export NCCL_IB_TC=16
#export NCCL_IB_SL=5
#export NCCL_IB_GID_INDEX=3
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=mlx5
#export NCCL_IB_TIMEOUT=22
#export NCCL_IB_QPS_PER_CONNECTION=8
#export NCCL_MIN_NCHANNELS=4
#export NCCL_NET_PLUGIN=none
export NCCL_P2P_DISABLE=1  # 禁用点对点通信
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand 支持
export GLOO_SOCKET_IFNAME=enp2s0f0np0  # 同步修正 GLOO 网卡
export NCCL_P2P_DISABLE=1  # 禁用点对点通信

# 从魔塔下载
export VLLM_USE_MODELSCOPE=True

MODEL_PATH=${1:-Qwen/Qwen3-4B-Instruct}
PORT=${2:-8000}

CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH"  \
  --port "$PORT" \
  --served-model-name "$MODEL_PATH" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32000

# bash run_vllm_server.sh Qwen/Qwen3-VL-4B-Instruct 8012 
#   --tensor-parallel-size 2 \