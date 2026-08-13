#!/usr/bin/env bash
set -euo pipefail

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10-trig-uniform-dual \
    --data=../datasets/cifar10.zip \
    --preset=cifar10-trig-uniform-dual \
    --precision=bf16 \
    --status=1100 \
    --snapshot=45000 \
    --checkpoint=45000 \
    --metrics=45000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=64 \
    "$@"
