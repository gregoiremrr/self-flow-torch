# Self-Flow on CIFAR-10

This project trains class-conditional pixel-space flow models on CIFAR-10.
It provides three controlled experiments:

1. Vanilla linear flow matching with one timestep per image.
2. Dual-Timestep Scheduling with two timesteps distributed across patches.
3. Full [Self-Flow](https://arxiv.org/abs/2603.06507) with an EMA teacher and
   tokenwise representation alignment.

Training uses four GPUs, BF16, a global batch of 256, a modern SiT backbone,
online image-quality evaluation, fixed-decay EMA, and W&B monitoring.

## Model

Images are represented directly as `32x32x3` pixels in `[-1, 1]`. The SiT
configuration is:

- Patch size: `2x2`
- Patch tokens: `16x16 = 256`
- Hidden width: 512
- Transformer depth: 12
- Attention heads: 8
- MLP ratio: 4
- Conditioning: per-token adaLN-Zero
- Transformer components: RMSNorm, SwiGLU, 2D RoPE, and q/k normalization
- Backbone parameters: 57.725M
- Full Self-Flow student parameters: 58.775M

The Self-Flow parameter increase comes from a trainable
`512 -> 1024 -> 512` SiLU projection head.

## Flow objective

The probability path runs from Gaussian noise at `t=0` to data at `t=1`:

```text
z_t = t * x_data + (1 - t) * x_noise
v_target = x_data - x_noise
```

The network can directly predict either the clean image `x` or velocity `v`.
The CIFAR-10 presets use direct velocity prediction and velocity-space loss:

```text
v_pred   = network(z_t, t)
v_target = x_data - x_noise
```

Clean-image prediction remains available through `--pred=x`. In that mode,
both prediction and target use `max(1 - t, 0.05)` as their denominator.

## Dual-Timestep Scheduling

For every training image:

1. Sample two independent timesteps `t,s ~ Uniform(0,1)`.
2. Sample an independent Bernoulli mask over the 256 patch tokens.
3. Assign `s` to 25% of tokens and `t` to the other 75%.
4. Corrupt every patch using its assigned timestep.
5. Condition each patch token on that same timestep through adaLN-Zero.

The two timesteps are not sorted. Consequently, each token has the uniform
marginal training distribution, while tokens within an image can have
different noise levels. Sampling uses homogeneous timesteps for every patch.

## Self-Flow objective

The student receives the mixed-timestep image. The teacher receives a
homogeneously corrupted image at the cleaner timestep:

```text
t_teacher = max(t, s)
```

The student and teacher use the same Gaussian noise and the same class-drop
decision. The teacher is a stop-gradient copy of the student updated after
each optimizer step:

```text
teacher = 0.9999 * teacher + 0.0001 * student
```

There is no teacher-EMA warmup. Student features from layer 4 pass through the
projection head and are compared with raw teacher features from layer 8. For
student token `q` and teacher token `z`, the representation loss is:

```text
L_rep = mean(1 - cosine_similarity(q, z))
L = L_flow + 0.8 * L_rep
```

Cosine similarity is computed over the hidden dimension and averaged over
all images and patch tokens.

## Repository layout

```text
.
├── train.py                   Training presets and command-line interface
├── generate_images.py         Distributed image generation
├── calculate_metrics.py       FID, FD-DINOv2, MIND, and MIND-DINOv2
├── reconstruct_phema.py       Post-hoc power-function EMA reconstruction
├── dataset_tool.py            Dataset conversion and packing
├── training/
│   ├── training_loop.py       Distributed optimization and evaluation loop
│   ├── model.py               Flow parameterization and Heun sampler
│   ├── loss.py                Flow, dual-timestep, and Self-Flow objectives
│   ├── networks.py            SiT and supporting network layers
│   ├── encoders.py            Pixel and VAE encoders
│   ├── schedulers.py          Learning-rate schedules
│   ├── ema.py                 Fixed, traditional, and power-function EMA
│   ├── monitoring.py          W&B logging
│   └── dataset.py             Folder and ZIP dataset reader
├── scripts/
│   ├── training/              Four-GPU experiment launchers
│   └── metrics/               Reference and offline metric launchers
├── training-runs/             Checkpoints, snapshots, logs, and W&B runs
└── out/                       Generated image directories
```

Large reusable files are placed beside the repository:

```text
../datasets/cifar10.zip
../fid-refs/cifar10.pkl
```

## Environment setup

The tested environment uses Linux, Python 3.13, CUDA 12.8, PyTorch 2.11, and
four NVIDIA A100 40GB GPUs.

For a fresh local virtual environment:

```bash
bash scripts/module.sh
source .venv/bin/activate
```

On the configured training machine:

```bash
source ../.venv/bin/activate
```

## Dataset preparation

Training expects the packed dataset at `../datasets/cifar10.zip`. Create it
from the official torchvision dataset with:

```bash
mkdir -p ../downloads ../raw-cifar10 ../datasets

python - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="../downloads", train=True, download=True)
root = Path("../raw-cifar10")
for index, (image, label) in enumerate(dataset):
    path = root / str(label) / f"{index:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
PY

python dataset_tool.py convert \
    --source=../raw-cifar10 \
    --dest=../datasets/cifar10.zip \
    --resolution=32x32
```

The resulting ZIP contains all 50,000 CIFAR-10 training images and class
labels.

## Metric reference preparation

Create the reference directory and compute all four reference statistics:

```bash
mkdir -p ../fid-refs
bash scripts/metrics/ref50k.sh
bash scripts/metrics/mind_ref5k.sh
```

The resulting `../fid-refs/cifar10.pkl` contains:

- FID Inception moments from 50,000 images
- FD-DINOv2 moments from 50,000 images
- MIND Inception features from 5,000 images
- MIND-DINOv2 features from 5,000 images

Detector weights are downloaded automatically on first use.

## Training

Activate the environment and launch one experiment from the repository root:

```bash
# Vanilla flow matching
bash scripts/training/script-cifar10.sh

# Dual-Timestep Scheduling without representation alignment
bash scripts/training/script-cifar10-dual.sh

# Full Self-Flow
bash scripts/training/script-cifar10-self-flow.sh
```

All launchers use four GPUs and a global batch of 256, giving 64 images per
GPU with one forward/backward round and no gradient accumulation.

The shared optimization configuration is:

- AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, and zero weight decay
- Dropout `0.13`
- Gradient clipping at norm `1.0`
- Linear warmup from 0 to `2e-4` over 10,000 steps
- Cosine learning-rate decay from `2e-4` to 0 over the remaining steps
- Fixed EMA decay `0.9999`
- Uniform training timesteps
- Adaptive timestep-dependent loss weighting

The vanilla and dual presets train for 500,000 optimizer steps. The Self-Flow
preset trains for 380,000 steps to account for its additional teacher forward
pass. These are compute-matched budgets of approximately 24 hours on four A100
GPUs, including periodic metrics.

The vanilla and dual launch intervals are optimizer-step counts:

- Status and sample grid: every 1,100 steps, approximately 3 minutes
- Snapshot, checkpoint, and online metrics: every 45,000 steps, approximately
  2 hours

The Self-Flow launcher uses 800 steps for status and 35,000 steps for
snapshots, checkpoints, and metrics.

Each launch creates a timestamped run directory, for example:

```text
training-runs/cifar10-vanilla/260811_125206_cifar10-vanilla
```

## Checkpoints and snapshots

A run directory contains:

- `training_options.json`: complete resolved configuration
- `log.txt`: console log
- `stats.jsonl`: machine-readable training statistics
- `training-state-*.pt`: resumable model, optimizer, EMA, and step state
- `model-snapshot-*.pkl`: fixed-decay EMA model

The same fixed-decay EMA is used for sample grids, online metrics, snapshots,
and the Self-Flow teacher. `training/ema.py` also provides
`TraditionalEMA` and `PowerFunctionEMA` for alternative experiments and
post-hoc EMA reconstruction. The final training step always writes a snapshot
and a training-state checkpoint.

Resume a run with:

```bash
bash scripts/training/script-cifar10-continue.sh \
    training-runs/cifar10-vanilla/<run-directory> \
    cifar10-vanilla
```

For a Self-Flow run, replace the final argument with `cifar10-self-flow`.

## Monitoring

W&B logs each scalar against optimizer steps, images processed, and wall-clock
hours. The main training signals are:

- Flow MSE
- Adaptive weighted flow loss
- Total loss
- `self_flow_loss`, equal to `mean(1 - cosine)`
- `feature_cosine`, equal to the mean cosine similarity
- Learning rate and gradient norm
- Timestep mean, standard deviation, and dual-timestep gap
- Mask fraction and teacher clean timestep
- Gradient-clipping coefficient and throughput

The media panels contain EMA sample grids and timestep histograms.

Online evaluations compute:

- FID with 20,000 generated images
- FD-DINOv2 with 20,000 generated images
- MIND with 5,000 generated images
- MIND-DINOv2 with 5,000 generated images

Use `WANDB_MODE=disabled` before a launch to perform a debugging run without
uploading W&B data.

## Offline generation and evaluation

Generate 50,000 images from a selected snapshot:

```bash
torchrun --standalone --nproc_per_node=4 generate_images.py \
    --outdir=out/cifar10-eval \
    --subdirs \
    --seeds=0-49999 \
    --model=training-runs/EXPERIMENT/RUN/model-snapshot-STEP.pkl \
    --sampler-fn=training.model.sample \
    --n-sampling-steps=50 \
    --guidance=1.0 \
    --max-batch-size=64 \
    --encoder-batch-size=64
```

Compute Fréchet metrics on all generated images:

```bash
python calculate_metrics.py calc \
    --images=out/cifar10-eval \
    --ref=../fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --num-images=50000 \
    --max-batch-size=64
```

Compute MIND metrics on 5,000 generated images:

```bash
python calculate_metrics.py calc \
    --images=out/cifar10-eval \
    --ref=../fid-refs/cifar10.pkl \
    --metrics=mind,mind_dinov2 \
    --num-images=5000 \
    --max-batch-size=64
```

## Configuration

`train.py` defines:

- `dataset_presets`: model, data scale, sampler, and learning-rate schedule
- `config_presets`: optimization and objective settings for each experiment

Command-line values override preset values. Run:

```bash
python train.py --help
```

to list every available option.

## References

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://arxiv.org/abs/2603.06507)
- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)
- [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)
- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)