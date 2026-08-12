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
    ARTIFACT_STEPS=35000
else
    STATUS_STEPS=1100
    ARTIFACT_STEPS=45000
fi

torchrun --standalone --nproc_per_node=4 train.py \
    --outdir="${RUN_DIR}" \
    --data=../datasets/cifar10.zip \
    --preset="${PRESET}" \
    --precision=bf16 \
    --status="${STATUS_STEPS}" \
    --snapshot="${ARTIFACT_STEPS}" \
    --checkpoint="${ARTIFACT_STEPS}" \
    --metrics="${ARTIFACT_STEPS}" \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=64
