import torch
import torch.nn.functional as F
from torch_utils import persistence

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(
        self,
        p_uncond=0.1,
        time_distribution='uniform',
        time_mu=0.0,
        time_sigma=1.0,
        dual_timestep=False,
        mask_ratio=0.25,
        self_flow=False,
        self_flow_weight=0.8,
        student_layer=None,
        teacher_layer=None,
    ):
        """
        Args:
            p_uncond: The probability of dropping the class label to train the unconditional branch.
        """
        if time_distribution not in ['uniform', 'logit_normal']:
            raise ValueError(f'Unknown time distribution: {time_distribution}')
        if not 0 <= mask_ratio <= 0.5:
            raise ValueError('Self-Flow mask_ratio must be in [0, 0.5]')
        if self_flow and not dual_timestep:
            raise ValueError('Self-Flow requires dual_timestep=True')
        self.p_uncond = p_uncond
        self.time_distribution = time_distribution
        self.time_mu = float(time_mu)
        self.time_sigma = float(time_sigma)
        self.dual_timestep = bool(dual_timestep)
        self.mask_ratio = float(mask_ratio)
        self.self_flow = bool(self_flow)
        self.self_flow_weight = float(self_flow_weight)
        self.student_layer = student_layer
        self.teacher_layer = teacher_layer

    def _sample_t(self, batch_size, device):
        if self.time_distribution == 'uniform':
            return torch.rand(batch_size, device=device)
        logits = (
            torch.randn(batch_size, device=device) * self.time_sigma
            + self.time_mu
        )
        return torch.sigmoid(logits)

    def __call__(self, model, images, labels=None, teacher_model=None):

        if labels is not None and self.p_uncond > 0.0:
            drop_mask = torch.rand(labels.shape[0], 1, device=labels.device) < self.p_uncond
            labels = torch.where(drop_mask, torch.zeros_like(labels), labels)

        # We assume that the model is wrapped in DDP.
        module = model.module
        batch_size = images.shape[0]
        t = self._sample_t(batch_size, images.device)
        secondary_t = None
        token_mask = None

        if self.dual_timestep:
            if not getattr(module.net, 'supports_per_token_t', False):
                raise RuntimeError('Dual-timestep training requires a per-token timestep network')
            num_tokens = module.net.x_embedder.num_patches
            secondary_t = self._sample_t(batch_size, images.device)
            # Do not sort t and s.  This is intentional: it preserves p(t) as
            # the marginal distribution of every token (Self-Flow Eq. 4).
            token_mask = (
                torch.rand(batch_size, num_tokens, device=images.device)
                < self.mask_ratio
            )
            model_t = torch.where(
                token_mask, secondary_t[:, None], t[:, None],
            )
        else:
            model_t = t

        time_map = module._time_to_image(model_t)
        x_noise = torch.randn_like(images) * module.sigma_data
        xt = time_map * images + (1.0 - time_map) * x_noise

        if module.pred == 'x':
            # Match JiT exactly: both target and prediction use the clipped
            # denominator when the clean image is the network output.
            denom = (1.0 - time_map).clamp_min(module.eps)
            v_target = (images - xt) / denom
        else:
            v_target = images - x_noise

        if self.self_flow:
            if teacher_model is None:
                raise RuntimeError('Self-Flow is enabled but no EMA teacher was provided')
            v_pred, logvar, student_features = model(
                xt,
                model_t,
                labels,
                return_logvar=True,
                return_features=True,
                feature_layer=self.student_layer,
                project_features=True,
            )
        else:
            v_pred, logvar = model(xt, model_t, labels, return_logvar=True)
            student_features = None

        # EDM2-style adaptive weighting (Karras et al., 2024).
        residual_sq = (v_pred - v_target).square()
        weighted_flow_loss = (torch.exp(-logvar) * residual_sq + logvar).mean()

        rep_loss = weighted_flow_loss.new_zeros([])
        feature_cosine = weighted_flow_loss.new_zeros([])
        teacher_clean_t = t
        if self.self_flow:
            # Our path runs noise -> data, opposite to the notation in the
            # Self-Flow paper, so max(t, s) is the cleaner teacher timestep.
            teacher_clean_t = torch.maximum(t, secondary_t)
            teacher_time_map = module._time_to_image(teacher_clean_t)
            teacher_xt = (
                teacher_time_map * images
                + (1.0 - teacher_time_map) * x_noise
            )
            with torch.no_grad():
                _, teacher_features = teacher_model(
                    teacher_xt,
                    teacher_clean_t,
                    labels,
                    return_features=True,
                    feature_layer=self.teacher_layer,
                    project_features=False,
                )
            feature_cosine = F.cosine_similarity(
                student_features.to(torch.float32),
                teacher_features.to(torch.float32),
                dim=-1,
            ).mean()
            # Maintainer-confirmed reduction: tokenwise cosine distance,
            # averaged over every token and sample.  The teacher branch is
            # stop-gradient by construction.
            rep_loss = 1.0 - feature_cosine

        total_loss = weighted_flow_loss + self.self_flow_weight * rep_loss

        # Side stats.
        time_values = model_t.detach()
        gap = (
            (t - secondary_t).abs()
            if secondary_t is not None
            else torch.zeros_like(t)
        )
        stats = dict(
            mse=residual_sq.mean().detach(),
            logvar=logvar.mean().detach(),
            weighted_flow_loss=weighted_flow_loss.detach(),
            rep_loss=rep_loss.detach(),
            feature_cosine=feature_cosine.detach(),
            time_mean=time_values.mean(),
            time_std=time_values.std(unbiased=False),
            dual_time_gap=gap.mean().detach(),
            masked_fraction=(
                token_mask.to(torch.float32).mean().detach()
                if token_mask is not None
                else weighted_flow_loss.new_zeros([])
            ),
            teacher_clean_time=teacher_clean_t.mean().detach(),
            time_samples=time_values.detach(),
            time_gap_samples=gap.detach(),
        )
        return total_loss, stats

#----------------------------------------------------------------------------
