import math
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import persistence
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Pixel normalization (Karras et al., 2017): RMS-norm across the channel dim.
# Used as `pnorm(.)` in the Adaptive Double Normalization variant of AdaGN.

def pixel_norm(x, eps=1e-8):
    # x: [B, C, ...] -> normalize along channel dim.
    return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(dim=1, keepdim=True) + eps).to(x.dtype)

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        adaptive_double_norm=False,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        # Adaptive Double Normalization is a drop-in replacement for AdaGN, so it
        # only makes sense when adaptive_scale is on (i.e. we have scale+shift).
        assert not adaptive_double_norm or adaptive_scale, \
            'adaptive_double_norm=True requires adaptive_scale=True'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.adaptive_double_norm = adaptive_double_norm

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            if self.adaptive_double_norm:
                # Adaptive Double Normalization: y = norm(x) * pnorm(s(t)) + pnorm(b(t)).
                # No "+1" offset on the scale here: pnorm already bounds its magnitude,
                # and the (+1) trick exists only to initialize AdaGN as the identity.
                scale = pixel_norm(scale)
                shift = pixel_norm(shift)
                x = silu(torch.addcmul(shift, self.norm1(x), scale))
            else:
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.

        adaptive_double_norm = False,       # Use Adaptive Double Normalization (pnorm'd AdaGN).
                                            # False = additive timestep embedding (default DDPM++/NCSN++).
                                            # True  = y = norm(x) * pnorm(s(t)) + pnorm(b(t)). Implies adaptive_scale=True.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        # Adaptive Double Normalization is a variant of AdaGN, so we must enable
        # the scale/shift path (adaptive_scale) whenever it is requested.
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True,
            adaptive_scale=adaptive_double_norm,
            adaptive_double_norm=adaptive_double_norm,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------

# Modern SiT/JiT building blocks.  The transformer keeps the adaLN-Zero
# conditioning used by SiT, and incorporates the generally useful JiT
# upgrades: RMSNorm, SwiGLU, 2D RoPE, and q/k normalization.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x * torch.rsqrt(x.to(torch.float32).square().mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return y * self.weight.to(x.dtype)


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10_000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        # Preserve every leading dimension.  In particular, [B, N] timesteps
        # become [B, N, D] embeddings for per-patch Self-Flow conditioning.
        shape = t.shape
        t = t.to(torch.float32).reshape(-1)
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = F.pad(emb, (0, 1))
        emb = self.mlp(emb)
        return emb.reshape(*shape, emb.shape[-1])


class PatchEmbed(torch.nn.Module):
    def __init__(self, img_resolution, patch_size, in_channels, hidden_size):
        super().__init__()
        assert img_resolution % patch_size == 0
        self.img_resolution = img_resolution
        self.patch_size = patch_size
        self.grid_size = img_resolution // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = torch.nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SwiGLUFFN(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # The 2/3 factor keeps SwiGLU's parameter count equal to a 4x GELU MLP.
        hidden_dim = int(hidden_dim * 2 / 3)
        hidden_dim += -hidden_dim % 8
        self.w12 = torch.nn.Linear(dim, 2 * hidden_dim)
        self.w3 = torch.nn.Linear(hidden_dim, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rope(x, cos, sin):
    cos = cos.to(device=x.device, dtype=x.dtype)[None, None]
    sin = sin.to(device=x.device, dtype=x.dtype)[None, None]
    return x * cos + _rotate_half(x) * sin


def _make_2d_rope(grid_size, head_dim, theta=10_000.0):
    assert head_dim % 4 == 0, '2D RoPE requires head_dim divisible by 4'
    axis_dim = head_dim // 2
    inv_freq = theta ** (
        -torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
    )
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing='ij',
    )

    def axis_embedding(position):
        angles = position.flatten()[:, None] * inv_freq[None]
        return (
            angles.cos().repeat_interleave(2, dim=-1),
            angles.sin().repeat_interleave(2, dim=-1),
        )

    cos_y, sin_y = axis_embedding(yy)
    cos_x, sin_x = axis_embedding(xx)
    return torch.cat([cos_y, cos_x], dim=-1), torch.cat([sin_y, sin_x], dim=-1)


def _get_1d_sincos_pos_embed(embed_dim, positions):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2
    omega = 1.0 / 10_000 ** omega
    angles = np.einsum('m,d->md', positions.reshape(-1), omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 4 == 0
    yy, xx = np.meshgrid(
        np.arange(grid_size, dtype=np.float64),
        np.arange(grid_size, dtype=np.float64),
        indexing='ij',
    )
    emb_y = _get_1d_sincos_pos_embed(embed_dim // 2, yy)
    emb_x = _get_1d_sincos_pos_embed(embed_dim // 2, xx)
    return np.concatenate([emb_y, emb_x], axis=1)


class SiTAttention(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, rope_cos, rope_sin):
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, num_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = _apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = _apply_rope(self.k_norm(k), rope_cos, rope_sin)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, num_tokens, dim)
        return self.proj_dropout(self.proj(x))


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SiTBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SiTAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), dropout=dropout,
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c, rope_cos, rope_sin):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_attn * self.attn(
            _modulate(self.norm1(x), shift_attn, scale_attn),
            rope_cos, rope_sin,
        )
        x = x + gate_mlp * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SiTFinalLayer(torch.nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = torch.nn.Linear(
            hidden_size, patch_size * patch_size * out_channels,
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class SelfFlowProjector(torch.nn.Module):
    def __init__(self, hidden_size, hidden_ratio=2.0):
        super().__init__()
        # Self-Flow's released SimpleHead is D -> 2D -> D with SiLU.
        inner = int(hidden_size * hidden_ratio)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, inner),
            torch.nn.SiLU(),
            torch.nn.Linear(inner, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


@persistence.persistent_class
class SiT(torch.nn.Module):
    """Pixel-space SiT with adaLN-Zero and per-patch timestep conditioning."""

    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim=0,
        patch_size=2,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        enable_projector=False,
        projector_hidden_ratio=2.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert (hidden_size // num_heads) % 4 == 0
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.supports_per_token_t = True

        self.x_embedder = PatchEmbed(
            img_resolution, patch_size, in_channels, hidden_size,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = (
            torch.nn.Linear(label_dim, hidden_size, bias=False)
            if label_dim > 0 else None
        )
        self.blocks = torch.nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final_layer = SiTFinalLayer(
            hidden_size, patch_size, out_channels,
        )
        self.projector = (
            SelfFlowProjector(hidden_size, projector_hidden_ratio)
            if enable_projector else None
        )

        pos_embed = _get_2d_sincos_pos_embed(
            hidden_size, self.x_embedder.grid_size,
        )
        rope_cos, rope_sin = _make_2d_rope(
            self.x_embedder.grid_size, hidden_size // num_heads,
        )
        self.register_buffer(
            'pos_embed',
            torch.from_numpy(pos_embed).to(torch.float32).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer('rope_cos', rope_cos, persistent=True)
        self.register_buffer('rope_sin', rope_sin, persistent=True)
        self.initialize_weights()

    def initialize_weights(self):
        def init_module(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(init_module)
        patch_weight = self.x_embedder.proj.weight
        torch.nn.init.xavier_uniform_(patch_weight.view(patch_weight.shape[0], -1))
        torch.nn.init.zeros_(self.x_embedder.proj.bias)

        for layer in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            torch.nn.init.normal_(layer.weight, std=0.02)
        if self.y_embedder is not None:
            torch.nn.init.normal_(self.y_embedder.weight, std=0.02)

        # adaLN-Zero makes every residual block and the output head an identity
        # or zero map at initialization, which is critical for stable SiT runs.
        for block in self.blocks:
            torch.nn.init.zeros_(block.adaLN_modulation[-1].weight)
            torch.nn.init.zeros_(block.adaLN_modulation[-1].bias)
        torch.nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        torch.nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        torch.nn.init.zeros_(self.final_layer.linear.weight)
        torch.nn.init.zeros_(self.final_layer.linear.bias)

    def _unpatchify(self, x):
        batch = x.shape[0]
        grid = self.x_embedder.grid_size
        patch = self.patch_size
        x = x.reshape(
            batch, grid, grid, patch, patch, self.out_channels,
        )
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(
            batch, self.out_channels,
            grid * patch, grid * patch,
        )

    def forward(
        self,
        x,
        noise_labels,
        class_labels=None,
        return_features=False,
        feature_layer=None,
        project_features=False,
        features_only=False,
    ):
        x = self.x_embedder(x) + self.pos_embed.to(x.dtype)
        batch, num_tokens, _ = x.shape

        t_emb = self.t_embedder(noise_labels)
        if t_emb.ndim == 2:
            t_emb = t_emb[:, None, :].expand(-1, num_tokens, -1)
        elif t_emb.ndim == 3:
            if t_emb.shape[1] != num_tokens:
                raise ValueError(
                    f'Expected {num_tokens} patch timesteps, got {t_emb.shape[1]}'
                )
        else:
            raise ValueError(
                f'Timesteps must have shape [B] or [B, N], got {tuple(noise_labels.shape)}'
            )

        if self.y_embedder is not None:
            if class_labels is None:
                class_labels = torch.zeros(
                    batch, self.label_dim,
                    device=x.device, dtype=x.dtype,
                )
            y_emb = self.y_embedder(class_labels.to(x.dtype))
            c = t_emb + y_emb[:, None, :]
        else:
            c = t_emb

        if features_only and not return_features:
            raise ValueError('features_only=True requires return_features=True')
        if return_features:
            if feature_layer is None:
                raise ValueError('feature_layer is required when return_features=True')
            if not 1 <= int(feature_layer) <= self.depth:
                raise ValueError(
                    f'feature_layer must be in [1, {self.depth}], got {feature_layer}'
                )

        features = None
        for layer_idx, block in enumerate(self.blocks, start=1):
            x = block(x, c, self.rope_cos, self.rope_sin)
            if return_features and layer_idx == int(feature_layer):
                features = x
                if features_only:
                    if project_features:
                        if self.projector is None:
                            raise RuntimeError('Self-Flow projector is not enabled')
                        features = self.projector(features)
                    return features

        output = self._unpatchify(self.final_layer(x, c))
        if not return_features:
            return output

        if project_features:
            if self.projector is None:
                raise RuntimeError('Self-Flow projector is not enabled')
            features = self.projector(features)
        return output, features

#----------------------------------------------------------------------------
