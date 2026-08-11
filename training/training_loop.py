import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb
from training import monitoring
from training import evaluation

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs,
    encoder_kwargs,
    data_loader_kwargs,
    model_kwargs,
    loss_kwargs,
    optimizer_kwargs,
    lr_kwargs,
    ema_kwargs,
    self_flow_ema_kwargs,
    sampler_kwargs,
    pretrained_pkl,
    max_clip_norm,

    run_dir,                # Output directory.
    seed,                   # Global random seed.
    batch_size,             # Total batch size for one training iteration.
    max_batch_gpu,          # Limit batch size per GPU. None = no limit.
    total_nimg,             # Train for a total of N training images.
    status_nimg,            # Report status every N training images. None = disable.
    snapshot_nimg,          # Save model snapshot every N training images. None = disable.
    checkpoint_nimg,        # Save state checkpoint every N training images. None = disable.
    metrics_nimg,           # Compute eval metrics every N training images. None = disable.
    metrics_kwargs,         # dict(metrics, ref_path, num_samples, mind_num_samples, max_batch_size). Required if metrics_nimg is set.

    loss_scaling,           # Loss scaling factor for reducing FP16 under/overflows.
    cudnn_benchmark,        # Enable torch.backends.cudnn.benchmark?
    force_finite,           # Get rid of NaN/Inf gradients before feeding them to the optimizer.
):
    # Device.
    device = torch.device('cuda')

    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if max_batch_gpu is None or max_batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    else:
        batch_gpu = max_batch_gpu

    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    dist.print0(f'Batch size: total {batch_size}, per-GPU {batch_gpu_total} '
                f'(micro-batch {batch_gpu} x {num_accumulation_rounds} accumulation rounds), '
                f'GPUs {dist.get_world_size()}')
    assert total_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or snapshot_nimg % batch_size == 0
    assert checkpoint_nimg is None or checkpoint_nimg % batch_size == 0
    assert metrics_nimg is None or metrics_nimg % batch_size == 0
    if metrics_nimg is not None:
        assert metrics_kwargs is not None and metrics_kwargs.get('ref_path'), \
            '--metrics requires --metric-ref to be set'

    # Setup dataset and encoder.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))

    # Setup the model, eventually loading the model.
    dist.print0('Constructing model...')
    interface_kwargs = dict(
        img_resolution=ref_image.shape[-1],
        img_channels=ref_image.shape[1],
        label_dim=ref_label.shape[-1]
    )
    model = dnnlib.util.construct_class_by_name(**model_kwargs, **interface_kwargs)
    if pretrained_pkl is not None:
        dist.print0(f'Loading pretrained weights from {pretrained_pkl}...')
        with open(pretrained_pkl, 'rb') as f:
            data = pickle.load(f)
        # data.ema is the saved model
        misc.copy_params_and_buffers(src_module=data.ema, dst_module=model, require_all=False)
        del data
    model.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        with torch.no_grad():
            misc.print_module_summary(model, [
                torch.zeros([batch_gpu, model.img_channels, model.img_resolution, model.img_resolution], device=device),
                torch.ones([batch_gpu], device=device),
                torch.zeros([batch_gpu, model.label_dim], device=device),
            ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, cur_step=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=model.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(model=model, **ema_kwargs) if ema_kwargs is not None else None
    self_flow_ema = (
        dnnlib.util.construct_class_by_name(model=model, **self_flow_ema_kwargs)
        if self_flow_ema_kwargs is not None else None
    )

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(
        state=state,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ema=ema,
        self_flow_ema=self_flow_ema,
    )
    checkpoint.load_latest(run_dir)
    assert total_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {total_nimg // 1000} kimg:')
    dist.print0()

    # Setup WandB (rank 0 only).
    wandb_run = None
    if dist.get_rank() == 0:
        if not state.get('wandb_run_id', None):
            state.wandb_run_id = wandb.util.generate_id()
        wandb_run = wandb.init(
            project='flow-matching',
            name=os.path.basename(run_dir),
            dir=run_dir,
            id=state.wandb_run_id,
            resume='allow',
        )
        monitoring.setup_wandb_metrics(wandb)

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
        start_idx=state.cur_nimg
    )
    dataset_iterator = iter(
        dnnlib.util.construct_class_by_name(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs
        )
    )
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    start_step = state.cur_step
    stats_jsonl = None
    step_stats = dnnlib.EasyDict()

    while True:
        done = (state.cur_nimg >= total_nimg)

        # Report status. Also force one report right after the very first
        # training step so we can sanity-check the run without waiting for a
        # full status tick.
        first_step_report = (state.cur_step == start_step + 1)
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0 or first_step_report) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            sec_per_tick = cur_time - prev_status_time
            sec_per_kimg = cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # W&B logging (rank 0 only).
            if wandb_run is not None:
                grid_np = None
                if ema is not None:
                    ema_model = ema.get()
                    if isinstance(ema_model, list):
                        ema_model = ema_model[0][0]
                    ema_model.eval()
                    grid = monitoring.generate_sample_grid(
                        ema_model, encoder, sampler_kwargs,
                        n_samples=16,
                        label_dim=model.label_dim,
                        # Cycle the seed by status-tick index so consecutive
                        # ticks use different noise/labels but every 20 ticks
                        # we revisit the same ones (handy for visual diffs).
                        seed=(state.cur_nimg // status_nimg) % 20,
                        device=device,
                    )
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()

                main_metrics = {
                    'flow_mse': step_stats.get('flow_mse', float('nan')),
                    'total_loss': step_stats.get('total_loss', float('nan')),
                    'feature_cosine': step_stats.get('feature_cosine', float('nan')),
                    'lr': step_stats.get('lr', float('nan')),
                    'grad_norm': step_stats.get('grad_norm', float('nan')),
                }
                metrics = {
                    'weighted_flow_loss': step_stats.get('weighted_flow_loss', float('nan')),
                    'self_flow_loss': step_stats.get('self_flow_loss', float('nan')),
                    'logvar': step_stats.get('logvar', float('nan')),
                    'time_mean': step_stats.get('time_mean', float('nan')),
                    'time_std': step_stats.get('time_std', float('nan')),
                    'dual_time_gap': step_stats.get('dual_time_gap', float('nan')),
                    'masked_fraction': step_stats.get('masked_fraction', float('nan')),
                    'teacher_clean_time': step_stats.get('teacher_clean_time', float('nan')),
                    'clip_coef': step_stats.get('clip_coef', float('nan')),
                    'sec_per_tick': sec_per_tick,
                    'sec_per_kimg': sec_per_kimg,
                }
                plot_caption = (
                    f"nimg: {state.cur_nimg}, "
                    f"nstep: {state.cur_step}, "
                    f"ntime: {dnnlib.util.format_time(state.total_elapsed_time)} "
                    f"({int(state.total_elapsed_time)}s)"
                )
                main_plots = {'ema_samples_50step': wandb.Image(grid_np, caption=plot_caption)} if grid_np is not None else None
                plots = {}
                if step_stats.get('time_samples', None) is not None:
                    plots['timestep_distribution'] = wandb.Histogram(
                        step_stats.time_samples,
                    )
                if step_stats.get('time_gap_samples', None) is not None:
                    plots['dual_timestep_gap'] = wandb.Histogram(
                        step_stats.time_gap_samples,
                    )

                monitoring.log_to_wandb(
                    wandb,
                    cur_step=state.cur_step,
                    cur_nimg=state.cur_nimg,
                    elapsed_time=state.total_elapsed_time,
                    main_metrics=main_metrics,
                    metrics=metrics,
                    main_plots=main_plots,
                    plots=plots or None,
                )

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, total_nimg // 1000)
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Compute eval metrics (FID / FD-DINOv2) on the current EMA model.
        # Runs on every rank because the feature accumulation is distributed,
        # but only rank 0 ends up with the final scalars and logs them to W&B.
        if (metrics_nimg is not None
                and (done or state.cur_nimg % metrics_nimg == 0)
                and state.cur_nimg != start_nimg):
            if ema is not None:
                ema_model = ema.get()
                if isinstance(ema_model, list):
                    ema_model = ema_model[0][0]
                ema_model.eval()

                metric_start = time.time()
                # Distinguish per-metric sample budgets in the log line.
                metric_names = list(metrics_kwargs['metrics'])
                fid_n = metrics_kwargs['num_samples']
                mind_n = metrics_kwargs.get('mind_num_samples', 5000)
                has_frechet = any(m in ('fid', 'fd_dinov2') for m in metric_names)
                has_mind = any(m in ('mind', 'mind_dinov2') for m in metric_names)
                samples_desc = []
                if has_frechet:
                    samples_desc.append(f'{fid_n} for FID/FD-DINOv2')
                if has_mind:
                    samples_desc.append(f'{mind_n} for MIND')
                dist.print0(f'Computing metrics ({", ".join(metric_names)}) on '
                            + ', '.join(samples_desc) + ' samples...')
                metric_results = evaluation.compute_metrics(
                    model=ema_model,
                    encoder=encoder,
                    sampler_kwargs=sampler_kwargs,
                    ref_path=metrics_kwargs['ref_path'],
                    num_samples=metrics_kwargs['num_samples'],
                    mind_num_samples=metrics_kwargs.get('mind_num_samples', 5000),
                    metrics=metrics_kwargs['metrics'],
                    max_batch_size=metrics_kwargs['max_batch_size'],
                    seed=0, # Fixed seed so each FID tick uses the same noise/labels
                    device=device,
                )
                metric_elapsed = time.time() - metric_start

                if dist.get_rank() == 0 and metric_results is not None:
                    msg = ', '.join(f'{k}={v:g}' for k, v in metric_results.items())
                    dist.print0(f'Metrics @ kimg {state.cur_nimg/1e3:.1f}: {msg} '
                                f'(took {metric_elapsed:.1f}s)')
                    if wandb_run is not None:
                        monitoring.log_to_wandb(
                            wandb,
                            cur_step=state.cur_step,
                            cur_nimg=state.cur_nimg,
                            elapsed_time=state.total_elapsed_time,
                            main_eval_metrics=metric_results,
                            metrics={'metric_eval_sec': metric_elapsed},
                        )
                # Don't count the eval time as training time on the next tick.
                prev_status_time = time.time()

        # Save model snapshot.
        if snapshot_nimg is not None and (done or state.cur_nimg % snapshot_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(model) if hasattr(optimizer, 'get_ema') else model
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_model, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_model).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'model-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(model)

        # Done?
        if done:
            break

        # Evaluate the flow and optional Self-Flow losses.
        step_stats.flow_mse = 0
        step_stats.total_loss = 0
        step_stats.weighted_flow_loss = 0
        step_stats.self_flow_loss = 0
        step_stats.feature_cosine = 0
        step_stats.logvar = 0
        step_stats.time_mean = 0
        step_stats.time_std = 0
        step_stats.dual_time_gap = 0
        step_stats.masked_fraction = 0
        step_stats.teacher_clean_time = 0
        step_stats.time_samples = None
        step_stats.time_gap_samples = None
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_step)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))

                total_loss, loss_stats = loss_fn(
                    model=ddp,
                    images=images,
                    labels=labels.to(device),
                    teacher_model=self_flow_ema.get() if self_flow_ema is not None else None,
                )

                training_stats.report('Loss/flow_mse', loss_stats['mse'])
                training_stats.report('Loss/weighted_flow', loss_stats['weighted_flow_loss'])
                training_stats.report('Loss/self_flow', loss_stats['rep_loss'])
                training_stats.report('Loss/feature_cosine', loss_stats['feature_cosine'])
                step_stats.flow_mse += loss_stats['mse'].item() / num_accumulation_rounds
                step_stats.total_loss += total_loss.item() / num_accumulation_rounds
                step_stats.weighted_flow_loss += loss_stats['weighted_flow_loss'].item() / num_accumulation_rounds
                step_stats.self_flow_loss += loss_stats['rep_loss'].item() / num_accumulation_rounds
                step_stats.feature_cosine += loss_stats['feature_cosine'].item() / num_accumulation_rounds
                step_stats.logvar += loss_stats['logvar'].item() / num_accumulation_rounds
                step_stats.time_mean += loss_stats['time_mean'].item() / num_accumulation_rounds
                step_stats.time_std += loss_stats['time_std'].item() / num_accumulation_rounds
                step_stats.dual_time_gap += loss_stats['dual_time_gap'].item() / num_accumulation_rounds
                step_stats.masked_fraction += loss_stats['masked_fraction'].item() / num_accumulation_rounds
                step_stats.teacher_clean_time += loss_stats['teacher_clean_time'].item() / num_accumulation_rounds
                next_nimg = state.cur_nimg + batch_size
                capture_time_plots = (
                    dist.get_rank() == 0
                    and status_nimg is not None
                    and (
                        next_nimg % status_nimg == 0
                        or state.cur_step == start_step
                        or next_nimg >= total_nimg
                    )
                )
                if capture_time_plots:
                    step_stats.time_samples = loss_stats['time_samples'].to(torch.float32).flatten().cpu().numpy()
                    step_stats.time_gap_samples = loss_stats['time_gap_samples'].to(torch.float32).flatten().cpu().numpy()

                loss = total_loss * (loss_scaling / num_accumulation_rounds)
                loss.backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Unscale the gradients
        inv_scale = 1 / loss_scaling
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(inv_scale)

                if force_finite:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

        # Clip gradients. A value of `max_clip_norm <= 0` (or None) disables
        # clipping but we still compute the grad norm for logging by passing
        # max_norm=inf (the in-place rescale becomes a no-op).
        clip_norm = max_clip_norm if (max_clip_norm is not None and max_clip_norm > 0) else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        clip_coef = min(1.0, clip_norm / (grad_norm.item() + 1e-12))

        optimizer.step()

        step_stats.lr = lr
        step_stats.grad_norm = grad_norm.item()
        step_stats.clip_coef = clip_coef

        # Update EMA and training state.
        state.cur_nimg += batch_size
        state.cur_step += 1
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        if self_flow_ema is not None:
            self_flow_ema.update()
        cumulative_training_time += time.time() - batch_start_time

    if dist.get_rank() == 0 and wandb_run is not None:
        wandb.finish()

#----------------------------------------------------------------------------
