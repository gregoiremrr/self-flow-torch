#!/usr/bin/env bash
set -euo pipefail

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

RUN_DIR=${1:?Usage: script-cifar10-continue.sh RUN_DIR [PRESET]}
PRESET=${2:-cifar10-vanilla}

# Intervals below are optimizer-step counts, not numbers of seen images.
if [[ "${PRESET}" == "cifar10-self-flow" ]]; then
    STATUS_STEPS=800
    SNAPSHOT_STEPS=50000
    CHECKPOINT_STEPS=150000
else
    STATUS_STEPS=1100
    SNAPSHOT_STEPS=65000
    CHECKPOINT_STEPS=195000
fi

torchrun --standalone --nproc_per_node=4 train.py \
    --outdir="${RUN_DIR}" \
    --data=../datasets/cifar10.zip \
    --preset="${PRESET}" \
    --precision=bf16 \
    --status="${STATUS_STEPS}" \
    --snapshot="${SNAPSHOT_STEPS}" \
    --checkpoint="${CHECKPOINT_STEPS}"
