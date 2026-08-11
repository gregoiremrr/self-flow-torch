import os
import time
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop
from calculate_metrics import parse_metric_list
import datetime

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------

def _wait_for_path(path, timeout=300, interval=0.1):
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f'Timed out after {timeout}s waiting for {path}')
        time.sleep(interval)

#----------------------------------------------------------------------------
# Dataset presets: things intrinsic to the data (network shape, noise scale,
# sampler used at monitoring time, learning-rate schedule family).

dataset_presets = {
    'cifar10': dnnlib.EasyDict(
        sigma_data=0.5,
        eps=0.05,
        phema_stds=[0.050, 0.100, 0.200],
        net_kwargs=dnnlib.EasyDict(
            class_name='training.networks.SiT',
            patch_size=2,
            hidden_size=512,
            depth=12,
            num_heads=8,
            mlp_ratio=4.0,
            projector_hidden_ratio=2.0,
        ),
        sampler_kwargs=dnnlib.EasyDict(
            func_name='training.model.sample',
            n_steps=50,
            guidance=1.0,
        ),
        lr_scheduler_kwargs=dnnlib.EasyDict(
            func_name='training.schedulers.warmup_constant_lr',
        ),
    ),
}

#----------------------------------------------------------------------------
# Configuration presets.  All three experiments share the same backbone,
# optimizer, timestep marginal, and batch size; only the two Self-Flow
# components differ.  This makes the requested ablation directly comparable.

_cifar10_base = dict(
    dataset='cifar10',
    cond=True,
    total_nsteps=500_000,
    batch_size=256,                 # 64 images/GPU on the requested 4 GPUs.
    pred='x',
    precision='bf16',
    t_scale=1000,
    p_uncond_labels=0.10,
    dropout=0.0,
    lr=1e-4,
    warmup_nsteps=1_000,
    max_clip_norm=1.0,
    time_distribution='uniform',   # Self-Flow ImageNet recipe; same p(t) in all runs.
    time_mu=-0.8,                  # Used only when --time-distribution=logit_normal.
    time_sigma=0.8,
    mask_ratio=0.25,
    self_flow_weight=0.8,
    student_layer=4,               # round(0.3 * depth) for depth=12.
    teacher_layer=8,               # round(0.7 * depth) for depth=12.
    teacher_decay=0.9999,
    optimizer_kwargs=dnnlib.EasyDict(
        class_name='torch.optim.AdamW',
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    ),
)

config_presets = {
    'cifar10-vanilla': dnnlib.EasyDict(
        **_cifar10_base,
        dual_timestep=False,
        self_flow=False,
    ),
    'cifar10-dual': dnnlib.EasyDict(
        **_cifar10_base,
        dual_timestep=True,
        self_flow=False,
    ),
    'cifar10-self-flow': dnnlib.EasyDict(
        **_cifar10_base,
        dual_timestep=True,
        self_flow=True,
    ),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='cifar10-vanilla', **opts):
    opts = dnnlib.EasyDict(opts)

    # Resolve presets.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    config_preset = config_presets[preset]

    dataset_name = config_preset['dataset']
    if dataset_name not in dataset_presets:
        raise click.ClickException(f'Invalid dataset preset "{dataset_name}"')
    dataset_preset = dataset_presets[dataset_name]

    # Sanity check: the two preset namespaces must not overlap.
    overlap = set(config_preset).intersection(dataset_preset)
    assert not overlap, f'config_preset and dataset_preset share keys: {sorted(overlap)}'

    # Merge presets, then apply CLI overrides (a CLI value of None means "use preset").
    merged = {**dataset_preset, **config_preset}
    for key, value in merged.items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Conditional vs unconditional sanity: no label dropout if not class-conditional.
    if not opts.cond:
        assert opts.p_uncond_labels == 0, '--p-uncond-labels must be 0 when --cond=False'
    if opts.self_flow and not opts.dual_timestep:
        raise click.ClickException('--self-flow requires --dual-timestep')

    c = dnnlib.EasyDict()

    # Dataset and Dataloader.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=True)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    c.data_loader_kwargs = dict(
        class_name='torch.utils.data.DataLoader',
        pin_memory=opts.pin_memory,
        num_workers=opts.num_workers,
        prefetch_factor=opts.prefetch_factor
    )

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.batch_size = opts.batch_size
    # Durations are specified in optimizer steps; the training loop and LR
    # scheduler work in image counts, so convert here (1 step == batch_size images).
    c.total_nimg = opts.total_nsteps * opts.batch_size
    c.model_kwargs = dnnlib.EasyDict(
        class_name='training.model.FlowMatchingModel',
        pred=opts.pred,
        sigma_data=opts.sigma_data,
        t_scale=opts.t_scale,
        eps=opts.eps,
        net_kwargs=dnnlib.EasyDict(
            **opts.net_kwargs,
            dropout=opts.dropout,
            enable_projector=opts.self_flow,
        ),
        precision=opts.precision,
    )
    c.ema_kwargs = dict(class_name='training.phema.PowerFunctionEMA', stds=list(opts.phema_stds))
    c.self_flow_ema_kwargs = (
        dnnlib.EasyDict(
            class_name='training.phema.FixedEMA',
            decay=opts.teacher_decay,
        )
        if opts.self_flow else None
    )
    c.loss_kwargs = dnnlib.EasyDict(
        class_name='training.loss.FlowMatchingLoss',
        p_uncond=opts.p_uncond_labels,
        time_distribution=opts.time_distribution,
        time_mu=opts.time_mu,
        time_sigma=opts.time_sigma,
        dual_timestep=opts.dual_timestep,
        mask_ratio=opts.mask_ratio,
        self_flow=opts.self_flow,
        self_flow_weight=opts.self_flow_weight,
        student_layer=opts.student_layer,
        teacher_layer=opts.teacher_layer,
    )
    c.optimizer_kwargs = dnnlib.EasyDict(**opts.optimizer_kwargs)
    c.lr_kwargs = dnnlib.EasyDict(
        **opts.lr_scheduler_kwargs,
        base_lr=opts.lr,
        total_nimg=c.total_nimg,
        warmup_nimg=opts.warmup_nsteps * opts.batch_size,
    )
    c.sampler_kwargs = dnnlib.EasyDict(**opts.sampler_kwargs)
    c.max_clip_norm = opts.max_clip_norm

    # Performance-related options.
    c.max_batch_gpu = opts.max_batch_gpu or None
    c.loss_scaling = opts.ls
    c.cudnn_benchmark = opts.bench
    c.force_finite = opts.force_finite

    # I/O-related options. (Intervals are given in optimizer steps.)
    c.status_nimg = opts.status * opts.batch_size if opts.status else None
    c.snapshot_nimg = opts.snapshot * opts.batch_size if opts.snapshot else None
    c.checkpoint_nimg = opts.checkpoint * opts.batch_size if opts.checkpoint else None

    # Eval metrics (FID / FD-DINOv2 / MIND). (Interval is given in optimizer steps.)
    c.metrics_nimg = opts.metrics * opts.batch_size if opts.metrics else None
    if c.metrics_nimg is not None:
        if not opts.metric_ref:
            raise click.ClickException('--metrics requires --metric-ref')
        # If metric_ref is a local path (not a URL), make sure it exists now
        # rather than failing after the first metric tick.
        if '://' not in opts.metric_ref and not os.path.isfile(opts.metric_ref):
            raise click.ClickException(f'--metric-ref: file not found: {opts.metric_ref}')
        c.metrics_kwargs = dnnlib.EasyDict(
            metrics=parse_metric_list(opts.metric_names),
            ref_path=opts.metric_ref,
            num_samples=opts.metric_num_samples,
            mind_num_samples=opts.mind_num_samples,
            max_batch_size=opts.metric_batch_size,
        )
    else:
        c.metrics_kwargs = None

    c.seed = opts.seed
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, pretrained_pkl, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Pretrained model:        {pretrained_pkl}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Precision:               {c.model_kwargs.precision}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, pretrained_pkl, c):
    options_path = os.path.join(run_dir, 'training_options.json')
    if dist.get_rank() == 0:
        if not os.path.isdir(run_dir):
            dist.print0('Creating output directory...')
            os.makedirs(run_dir)
        with open(options_path, 'wt') as f:
            json.dump(c, f, indent=2)
    else:
        # Wait until rank 0 has created the run directory and written the
        # training options file.
        _wait_for_path(options_path)

    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, pretrained_pkl=pretrained_pkl, **c)

#----------------------------------------------------------------------------
# Parse an integer (image count or optimizer-step count) with optional
# power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_count(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Output directory (resumed if exists with checkpoints)', metavar='DIR', type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--pretrained-pkl',   help='Pretrained snapshot path', metavar='DIR', type=str,   default=None)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='cifar10-vanilla', show_default=True)

# Hyperparameters. (None by default => use the preset value)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=None)
@click.option('--total_nsteps',     help='Training duration in optimizer steps', metavar='STEPS', type=parse_count, default=None)
@click.option('--batch-size',       help='Total batch size', metavar='NIMG',                    type=parse_count, default=None)
@click.option('--pred',             help='Quantity predicted by the network', metavar='x/v',    type=click.Choice(['x', 'v']), default=None)
@click.option('--precision',        help='Transformer compute precision', metavar='DTYPE',       type=click.Choice(['fp32', 'fp16', 'bf16']), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--t-scale',          help='Scaling for the t embedding', metavar='FLOAT',        type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--max_clip_norm',    help='Max gradient norm for clipping (0 disables clipping but still logs grad norm)', metavar='FLOAT', type=click.FloatRange(min=0), default=None)
@click.option('--p-uncond-labels',  help='Prob. of dropping labels for CFG training', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=None)
@click.option('--time-distribution', help='Training timestep marginal p(t)', metavar='DIST',     type=click.Choice(['uniform', 'logit_normal']), default=None)
@click.option('--time-mu',          help='Logit-normal timestep mean', metavar='FLOAT',          type=float, default=None)
@click.option('--time-sigma',       help='Logit-normal timestep std.', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--dual-timestep/--single-timestep', help='Noise patches at two iid timesteps',   default=None)
@click.option('--self-flow/--no-self-flow', help='Add EMA-teacher feature alignment',            default=None)
@click.option('--mask-ratio',       help='Fraction of tokens assigned the second timestep', metavar='FLOAT', type=click.FloatRange(min=0, max=0.5), default=None)
@click.option('--self-flow-weight', help='Representation loss coefficient gamma', metavar='FLOAT', type=click.FloatRange(min=0), default=None)
@click.option('--student-layer',    help='Student feature layer (1-indexed)', metavar='INT',     type=click.IntRange(min=1), default=None)
@click.option('--teacher-layer',    help='EMA teacher feature layer (1-indexed)', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--teacher-decay',    help='Fixed EMA teacher decay', metavar='FLOAT',            type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), default=None)

# Performance-related options.
@click.option('--max-batch-gpu',    help='Limit batch size per GPU (smaller values enable accumulation)', metavar='NIMG', type=parse_count, default=None, show_default=True)
@click.option('--pin-memory',       help='Use pinned memory in the dataloader', metavar='BOOL', default=True, show_default=True)
@click.option('--num-workers',      help='Number of workers in the dataloader', metavar='INT',  type=int, default=2, show_default=True)
@click.option('--prefetch_factor',  help='Number of batches for each worker', metavar='INT',    type=int, default=2, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)
@click.option('--force-finite',     help='Zero NaN/Inf gradients before optimizer step',        metavar='BOOL', type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints (optimizer steps)', metavar='STEPS',     type=parse_count, default='512', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots (optimizer steps)', metavar='STEPS', type=parse_count, default='32Ki', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints (optimizer steps)', metavar='STEPS', type=parse_count, default='512Ki', show_default=True)

# Eval-metrics-related options.
@click.option('--metrics',          help='Interval of FID/FD-DINOv2/MIND evaluation in optimizer steps. Disabled by default.', metavar='STEPS', type=parse_count, default=None, show_default=True)
@click.option('--metric-names',     help='Comma-separated list of metrics to compute (fid, fd_dinov2, mind, mind_dinov2)', metavar='LIST', type=str, default='fid', show_default=True)
@click.option('--metric-num-samples', help='Number of generated samples for Fr\u00e9chet metrics (fid, fd_dinov2)', metavar='INT', type=click.IntRange(min=2), default=20000, show_default=True)
@click.option('--mind-num-samples', help='Number of generated samples for MIND metrics (mind, mind_dinov2)', metavar='INT', type=click.IntRange(min=2), default=5000, show_default=True)
@click.option('--metric-ref',       help='Reference statistics .pkl/.npz', metavar='PATH', type=str, default='../fid-refs/cifar10.pkl', show_default=True)
@click.option('--metric-batch-size',help='Per-rank batch size for metric sampling/feature extraction', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)


def cmdline(outdir, pretrained_pkl, dry_run, **opts):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)

    # If outdir has no timestamp yet, add one.
    # If it already exists (user is resuming), use as-is.
    if os.path.isdir(outdir) and any(f.startswith('training-state-') for f in os.listdir(outdir)):
        run_dir = outdir
        dist.print0(f'Resuming from {run_dir}')

        if pretrained_pkl:
            raise click.ClickException('Cannot use --pretrained when resuming from an existing run')
    else:
        # Pick a fresh timestamped run_dir on rank 0 and share it via the filesystem.
        preset_name = opts.get('preset', 'run')
        os.makedirs(outdir, exist_ok=True)
        # Use the torchelastic run id (set per `torchrun` invocation) so the
        # marker file is unique to this launch and can be cleaned up safely.
        run_id = os.environ.get('TORCHELASTIC_RUN_ID', os.environ.get('MASTER_PORT', 'default'))
        marker_path = os.path.join(outdir, f'.run_dir.{run_id}')
        if dist.get_rank() == 0:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            run_dir = os.path.join(outdir, f'{now}_{preset_name}')
            with open(marker_path, 'wt') as f:
                f.write(run_dir)
        else:
            _wait_for_path(marker_path)
            with open(marker_path, 'rt') as f:
                run_dir = f.read().strip()

    print_training_config(run_dir=run_dir, pretrained_pkl=pretrained_pkl, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=run_dir, pretrained_pkl=pretrained_pkl, c=c)
    torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
