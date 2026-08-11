#!/usr/bin/env bash
set -euo pipefail

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

torchrun --standalone --nproc_per_node=4 calculate_metrics.py ref \
    --data=../datasets/cifar10.zip \
    --dest=../fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --num-images=50000 \
    --overwrite \
    --max-batch-size=64
