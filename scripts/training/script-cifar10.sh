#!/usr/bin/env bash
set -euo pipefail

# /etc/nccl.conf on GCE Deep Learning VMs forces NCCL_NET=gIB, which is
# Google's GPUDirect-RDMA networking stack for A3/H100 instances. On A2/A100
# (or any single-node run) gIB has no fabric to talk to and fails to init.
# Override here to use the built-in Socket transport for the control plane;
# NCCL still uses NVLink/P2P for actual GPU<->GPU traffic.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# Optimizer-step intervals calibrated from the 4xA100 vanilla run:
# 1100 steps ~3 min; 45000 steps ~2 h.
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10-vanilla \
    --data=../datasets/cifar10.zip \
    --preset=cifar10-vanilla \
    --precision=bf16 \
    --status=1100 \
    --snapshot=45000 \
    --checkpoint=45000 \
    --metrics=45000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=64
