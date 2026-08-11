#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10-dry-run \
    --data=../datasets/cifar10.zip \
    --preset=cifar10-self-flow \
    --precision=bf16 \
    --status=4 \
    --snapshot=256 \
    --checkpoint=512 \
    --dry-run
