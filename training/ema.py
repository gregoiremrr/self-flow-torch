# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/.

"""Exponential moving averages used for evaluation and Self-Flow training."""

import copy
import numpy as np
import torch

#----------------------------------------------------------------------------
# Convert power function exponent to relative standard deviation
# according to Equation 123 of EDM2.

def exp_to_std(exp):
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std

#----------------------------------------------------------------------------
# Convert relative standard deviation to power function exponent
# according to Equation 126 and Algorithm 2 of EDM2.

def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

#----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108 of EDM2.

def power_function_response(ofs, std, len, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equations 151 and Algorithm 3 of EDM2.

def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile according to Equation 127.

def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction.

def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std): # => [in, out]
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X

#----------------------------------------------------------------------------
# Power-function EMA from EDM2. It tracks scale-relative averaging profiles
# whose effective per-step decay changes as training progresses.

class PowerFunctionEMA:
    @torch.no_grad()
    def __init__(self, model, stds=[0.050, 0.100]):
        self.model = model
        self.stds = stds
        self.emas = [copy.deepcopy(model) for _std in stds]

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            for p_model, p_ema in zip(self.model.parameters(), ema.parameters()):
                p_ema.copy_(p_model)

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size)
            for p_model, p_ema in zip(self.model.parameters(), ema.parameters()):
                p_ema.lerp_(p_model, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for p_model, p_ema in zip(self.model.buffers(), ema.buffers()):
                p_ema.copy_(p_model)
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = state['stds']
        for ema, s_ema in zip(self.emas, state['emas']):
            ema.load_state_dict(s_ema)

#----------------------------------------------------------------------------
# Traditional half-life EMA. Its decay is specified in millions of images and
# can be shortened automatically during early training via ``rampup_ratio``.

class TraditionalEMA:
    @torch.no_grad()
    def __init__(self, model, halflife_Mimg=float('inf'), rampup_ratio=0.09):
        self.model = model
        self.halflife_Mimg = halflife_Mimg
        self.rampup_ratio = rampup_ratio
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def reset(self):
        for p_model, p_ema in zip(self.model.parameters(), self.ema.parameters()):
            p_ema.copy_(p_model)

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        halflife_Mimg = self.halflife_Mimg
        if self.rampup_ratio is not None:
            halflife_Mimg = min(halflife_Mimg, cur_nimg / 1e6 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_Mimg * 1e6, 1e-8))
        for p_model, p_ema in zip(self.model.parameters(), self.ema.parameters()):
            p_ema.lerp_(p_model, 1 - beta)
        for b_model, b_ema in zip(self.model.buffers(), self.ema.buffers()):
            b_ema.copy_(b_model)

    @torch.no_grad()
    def get(self):
        return self.ema

    def state_dict(self):
        return dict(
            halflife_Mimg=self.halflife_Mimg,
            rampup_ratio=self.rampup_ratio,
            ema=self.ema.state_dict(),
        )

    def load_state_dict(self, state):
        # Accept checkpoints from the previous implementation, which stored
        # only the averaged model state.
        if 'ema' not in state:
            self.ema.load_state_dict(state)
            return
        self.halflife_Mimg = state['halflife_Mimg']
        self.rampup_ratio = state['rampup_ratio']
        self.ema.load_state_dict(state['ema'])

#----------------------------------------------------------------------------
# Fixed per-optimizer-step EMA. This is the standard decay=0.9999 formulation
# used by DiT, JiT, and Self-Flow.

class FixedEMA:
    @torch.no_grad()
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, cur_nimg=None, batch_size=None):
        del cur_nimg, batch_size
        for p_model, p_ema in zip(self.model.parameters(), self.ema.parameters()):
            p_ema.lerp_(p_model, 1.0 - self.decay)
        for b_model, b_ema in zip(self.model.buffers(), self.ema.buffers()):
            b_ema.copy_(b_model)

    @torch.no_grad()
    def get(self):
        return self.ema

    def state_dict(self):
        return dict(decay=self.decay, ema=self.ema.state_dict())

    def load_state_dict(self, state):
        self.decay = float(state['decay'])
        self.ema.load_state_dict(state['ema'])

#----------------------------------------------------------------------------
