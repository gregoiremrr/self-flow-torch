import torch
from torch_utils import persistence
from training.networks import FourierEmbedding, Linear

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingModel(torch.nn.Module):
    def __init__(
        self,
        interpolant,
        img_resolution,
        img_channels,
        sigma_data,
        label_dim=0,
        t_scale=1000,
        precision='fp32',
        net_kwargs=None,
        logvar_channels=128,
    ):
        assert interpolant in ['linear', 'trig']
        assert precision in ['fp32', 'fp16', 'bf16']
        super().__init__()
        self.interpolant = interpolant
        self.sigma_data = sigma_data
        self.precision = precision
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.t_scale = t_scale
        if label_dim > 0:
            self.register_buffer('uncond_label', torch.zeros([1, label_dim]))
        else:
            self.uncond_label = None
        import dnnlib
        self.net = dnnlib.util.construct_class_by_name(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **net_kwargs,
        )
        # EDM2-style adaptive loss weighting (Karras et al., 2024).
        self.logvar_fourier = FourierEmbedding(num_channels=logvar_channels)
        self.logvar_linear = Linear(in_features=logvar_channels, out_features=1, init_mode='kaiming_normal')

    def _time_to_image(self, t):
        """Broadcast scalar or patch-token timesteps to a [B,1,H,W] map."""
        if t.ndim == 1:
            return t[:, None, None, None]
        if t.ndim != 2:
            raise ValueError(f'Expected timesteps with shape [B] or [B,N], got {tuple(t.shape)}')

        patch_size = getattr(self.net, 'patch_size', None)
        if patch_size is None:
            raise ValueError('Per-token timesteps require a patch-based network')
        grid = self.img_resolution // patch_size
        if t.shape[1] != grid * grid:
            raise ValueError(f'Expected {grid * grid} patch timesteps, got {t.shape[1]}')
        return (
            t.reshape(t.shape[0], grid, grid)
            .repeat_interleave(patch_size, dim=1)
            .repeat_interleave(patch_size, dim=2)
            .unsqueeze(1)
        )

    def interpolate(self, data, noise, t):
        """Construct z_t with noise at t=0 and data at t=1."""
        if self.interpolant == 'linear':
            return t * data + (1.0 - t) * noise
        angle = t * (torch.pi / 2)
        return torch.sin(angle) * data + torch.cos(angle) * noise

    def velocity_target(self, data, noise, t):
        """Return dz_t/dt for normalized t in [0, 1]."""
        if self.interpolant == 'linear':
            return data - noise
        angle = t * (torch.pi / 2)
        return (torch.pi / 2) * (
            torch.cos(angle) * data - torch.sin(angle) * noise
        )

    def forward(
        self,
        xt,
        t,
        class_labels=None,
        force_fp32=False,
        return_logvar=False,
        return_features=False,
        feature_layer=None,
        project_features=False,
        features_only=False,
    ):
        if features_only and (not return_features or return_logvar):
            raise ValueError(
                'features_only=True requires return_features=True and return_logvar=False'
            )
        xt = xt.to(torch.float32)
        t = t.to(torch.float32)
        t_scaled = t * self.t_scale

        autocast_enabled = (
            xt.device.type == 'cuda'
            and not force_fp32
            and self.precision != 'fp32'
        )
        autocast_dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        with torch.autocast(
            device_type='cuda',
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            feature_kwargs = {}
            if return_features:
                feature_kwargs = dict(
                    return_features=True,
                    feature_layer=feature_layer,
                    project_features=project_features,
                    features_only=features_only,
                )
            net_result = self.net(
                xt / self.sigma_data,
                t_scaled,
                class_labels=class_labels,
                **feature_kwargs,
            )

        if features_only:
            return net_result.to(torch.float32)
        if return_features:
            F_x, features = net_result
            features = features.to(torch.float32)
        else:
            F_x = net_result
            features = None

        # The network directly predicts dz_t/dt in data coordinates.
        v_pred = self.sigma_data * F_x.to(torch.float32)

        outputs = [v_pred]
        if return_logvar:
            flat_t = t.reshape(-1)
            logvar = self.logvar_linear(self.logvar_fourier(flat_t))
            logvar = logvar.reshape(*t.shape)
            outputs.append(self._time_to_image(logvar))
        if return_features:
            outputs.append(features)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def sample(model, labels, n_samples, n_steps, guidance=1.0, noise=None):
    """
    Sample from the model using a 2nd-order Heun solver with CFG support.
    Integrates dx/dt = v from noise at t=0 to data at t=1.
    """
    device = next(model.parameters()).device
    schedule = torch.linspace(0, 1, n_steps + 1, device=device)

    if noise is None:
        x = torch.randn(
            n_samples, model.img_channels, model.img_resolution, model.img_resolution,
            device=device,
        ) * model.sigma_data
    else:
        x = noise * model.sigma_data

    def get_guided_v(xt, t_cur, labels):
        if guidance == 1.0 or model.uncond_label is None:
            return model(xt, t_cur, class_labels=labels)

        # Batch the conditional and unconditional passes together for efficiency.
        xt_combined = torch.cat([xt, xt], dim=0)
        t_combined = torch.cat([t_cur, t_cur], dim=0)
        l_combined = torch.cat([labels, model.uncond_label.expand_as(labels)], dim=0)

        v_combined = model(xt_combined, t_combined, class_labels=l_combined)
        v_cond, v_uncond = v_combined.chunk(2)

        return torch.lerp(v_uncond, v_cond, guidance)

    with torch.no_grad():
        for i in range(n_steps):
            t_cur = schedule[i].expand(n_samples)
            t_next = schedule[i + 1].expand(n_samples)
            dt = schedule[i + 1] - schedule[i]

            # First evaluation (k1).
            k1 = get_guided_v(x, t_cur, labels)

            # Skip the Heun correction at the very last integration step.
            if i == n_steps - 1:
                x = x + dt * k1
            else:
                # Heun correction (2nd order).
                k2 = get_guided_v(x + dt * k1, t_next, labels)
                x = x + 0.5 * dt * (k1 + k2)

    return x

#----------------------------------------------------------------------------
