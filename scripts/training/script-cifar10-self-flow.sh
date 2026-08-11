#!/usr/bin/env bash
set -euo pipefail

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# Full Self-Flow: Dual-Timestep Scheduling plus EMA-teacher feature alignment.
# The teacher forward is estimated to add ~33% over the measured vanilla step.
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10-self-flow \
    --data=../datasets/cifar10.zip \
    --preset=cifar10-self-flow \
    --precision=bf16 \
    --status=800 \
    --snapshot=50000 \
    --checkpoint=150000 \
    --metrics=50000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=64
