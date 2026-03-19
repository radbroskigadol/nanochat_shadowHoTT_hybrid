"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

def _adamw_step_fused_impl(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """Rank-bucketed AdamW implementation to avoid mixed-rank compile cache invalidation."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def _optimizer_compile_enabled() -> bool:
    override = os.environ.get("NANOCHAT_OPTIM_COMPILE", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    return hasattr(torch, "compile") and torch.cuda.is_available()


def _maybe_compile_optimizer(fn: Callable[..., None], **kwargs: Any) -> Callable[..., None]:
    return torch.compile(fn, **kwargs) if _optimizer_compile_enabled() else fn


_adamw_step_fused_rank1 = _maybe_compile_optimizer(_adamw_step_fused_impl, dynamic=True, fullgraph=False)
_adamw_step_fused_rankN = _maybe_compile_optimizer(_adamw_step_fused_impl, dynamic=True, fullgraph=False)

def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    if p.ndim <= 1:
        _adamw_step_fused_rank1(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)
    else:
        _adamw_step_fused_rankN(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def _muon_step_fused_impl(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


muon_step_fused = _maybe_compile_optimizer(_muon_step_fused_impl, dynamic=False, fullgraph=True)


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-12 else 0.0


def _build_shadow_hyper_profile(report: dict[str, Any] | None, acceptance: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    acceptance = acceptance or {}
    hist = dict(report.get('four_value_histogram', {}))
    inv = dict(report.get('invariants', {}))
    control = dict(report.get('control', {}))
    truth = float(hist.get('T', 0.0))
    falsity = float(hist.get('F', 0.0))
    both = float(hist.get('B', 0.0))
    neither = float(hist.get('N', 0.0))
    contradiction = _clamp_float(both + neither, 0.0, 1.0)
    coherence_defect = _clamp_float(inv.get('coherence_defect', contradiction), 0.0, 1.0)
    provenance = _clamp_float(inv.get('provenance_coherence', 1.0 - coherence_defect), 0.0, 1.0)
    mutation = _clamp_float(inv.get('mutation_risk', contradiction), 0.0, 1.0)
    entropy = _clamp_float(inv.get('entropy', 0.0), 0.0, 1.0)
    margin = _clamp_float(inv.get('margin', 0.0), -1.0, 1.0)
    stability = _clamp_float(1.0 - 0.6 * coherence_defect - 0.4 * mutation, 0.0, 1.0)
    confidence = _clamp_float(0.5 * (truth + provenance) + 0.25 * max(0.0, margin), 0.0, 1.0)
    exploration = _clamp_float(0.5 * entropy + 0.5 * contradiction, 0.0, 1.0)
    accepted = float(acceptance.get('accepted_templates', 0.0) or 0.0)
    rejected = float(acceptance.get('rejected_templates', 0.0) or 0.0)
    acceptance_balance = _clamp_float(_safe_ratio(accepted - rejected, accepted + rejected + 1.0), -1.0, 1.0)
    return {
        'profile_id': report.get('profile_id', 'shadowhott-optimizer-v1'),
        'truth_mass': truth,
        'falsity_mass': falsity,
        'both_mass': both,
        'neither_mass': neither,
        'contradiction': contradiction,
        'coherence_defect': coherence_defect,
        'provenance': provenance,
        'mutation_risk': mutation,
        'entropy': entropy,
        'margin': margin,
        'stability': stability,
        'confidence': confidence,
        'exploration': exploration,
        'acceptance_balance': acceptance_balance,
        'control': control,
    }


def _shadow_group_adjustments(group: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {
            'lr_scale': 1.0,
            'wd_scale': 1.0,
            'momentum_scale': 1.0,
            'preservation_profile': 'base-update-family',
            'objective_profile': 'none',
            'witness': {},
        }
    role = str(group.get('role', group.get('kind', 'generic')))
    stability = float(profile['stability'])
    confidence = float(profile['confidence'])
    exploration = float(profile['exploration'])
    contradiction = float(profile['contradiction'])
    provenance = float(profile['provenance'])
    mutation = float(profile['mutation_risk'])
    acceptance_balance = float(profile['acceptance_balance'])
    if group.get('kind') == 'muon':
        lr_scale = _clamp_float(0.90 + 0.22 * confidence - 0.30 * exploration + 0.05 * acceptance_balance, 0.65, 1.15)
        wd_scale = _clamp_float(1.00 + 0.25 * (1.0 - stability) + 0.15 * contradiction, 0.85, 1.35)
        momentum_scale = _clamp_float(0.95 + 0.08 * stability - 0.10 * exploration, 0.82, 1.05)
    elif role == 'shadowhott':
        lr_scale = _clamp_float(0.95 + 0.30 * (1.0 - provenance) + 0.10 * exploration + 0.05 * acceptance_balance, 0.80, 1.35)
        wd_scale = _clamp_float(0.90 + 0.20 * stability, 0.75, 1.15)
        momentum_scale = 1.0
    elif role in {'embedding', 'value_embedding', 'lm_head'}:
        lr_scale = _clamp_float(0.96 + 0.10 * confidence - 0.18 * exploration, 0.82, 1.10)
        wd_scale = _clamp_float(1.00 + 0.08 * contradiction + 0.06 * (1.0 - provenance), 0.90, 1.15)
        momentum_scale = 1.0
    else:
        lr_scale = _clamp_float(0.92 + 0.14 * confidence - 0.20 * exploration + 0.03 * acceptance_balance, 0.78, 1.10)
        wd_scale = _clamp_float(1.00 + 0.15 * contradiction + 0.10 * mutation, 0.85, 1.25)
        momentum_scale = 1.0
    return {
        'lr_scale': lr_scale,
        'wd_scale': wd_scale,
        'momentum_scale': momentum_scale,
        'preservation_profile': 'preserve-update-family-and-parameter-assignment',
        'objective_profile': 'shadow-stability-vs-contradiction bounded hyperparameter modulation',
        'witness': {
            'role': role,
            'kind': group.get('kind', 'unknown'),
            'stability': stability,
            'confidence': confidence,
            'exploration': exploration,
            'contradiction': contradiction,
            'provenance': provenance,
            'mutation_risk': mutation,
            'acceptance_balance': acceptance_balance,
        },
    }

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict], shadow_report_getter: Callable[[], dict[str, Any]] | None = None, shadow_acceptance_getter: Callable[[], dict[str, Any]] | None = None):
        super().__init__(param_groups, defaults={})
        self._shadow_report_getter = shadow_report_getter
        self._shadow_acceptance_getter = shadow_acceptance_getter
        self._last_shadow_optimizer_report: dict[str, Any] = {}
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def set_shadow_context(self, shadow_report_getter: Callable[[], dict[str, Any]] | None = None, shadow_acceptance_getter: Callable[[], dict[str, Any]] | None = None) -> None:
        self._shadow_report_getter = shadow_report_getter
        self._shadow_acceptance_getter = shadow_acceptance_getter

    def get_last_shadow_optimizer_report(self) -> dict[str, Any]:
        return dict(self._last_shadow_optimizer_report)

    def _shadow_profile(self) -> dict[str, Any] | None:
        report = self._shadow_report_getter() if self._shadow_report_getter is not None else {}
        acceptance = self._shadow_acceptance_getter() if self._shadow_acceptance_getter is not None else {}
        if not report and not acceptance:
            return None
        return _build_shadow_hyper_profile(report, acceptance)

    def _group_adjustments(self, group: dict, profile: dict[str, Any] | None) -> dict[str, Any]:
        return _shadow_group_adjustments(group, profile)

    def _step_adamw(self, group: dict, adjustments: dict[str, Any] | None = None) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        adjustments = adjustments or {'lr_scale': 1.0, 'wd_scale': 1.0}
        effective_lr = float(group['lr']) * float(adjustments.get('lr_scale', 1.0))
        effective_wd = float(group['weight_decay']) * float(adjustments.get('wd_scale', 1.0))
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(effective_lr)
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(effective_wd)

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict, adjustments: dict[str, Any] | None = None) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        adjustments = adjustments or {'lr_scale': 1.0, 'wd_scale': 1.0, 'momentum_scale': 1.0}
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        effective_momentum = _clamp_float(float(group["momentum"]) * float(adjustments.get("momentum_scale", 1.0)), 0.0, 0.9999)
        effective_lr = float(group["lr"]) * float(adjustments.get("lr_scale", 1.0))
        effective_wd = float(group["weight_decay"]) * float(adjustments.get("wd_scale", 1.0))
        self._muon_momentum_t.fill_(effective_momentum)
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(effective_lr * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(effective_wd)

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        profile = self._shadow_profile()
        group_reports: list[dict[str, Any]] = []
        for group in self.param_groups:
            adjustments = self._group_adjustments(group, profile)
            if group['kind'] == 'adamw':
                self._step_adamw(group, adjustments)
            elif group['kind'] == 'muon':
                self._step_muon(group, adjustments)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
            group_reports.append({
                'role': group.get('role', group.get('kind', 'unknown')),
                'kind': group['kind'],
                'lr_scale': float(adjustments.get('lr_scale', 1.0)),
                'wd_scale': float(adjustments.get('wd_scale', 1.0)),
                'momentum_scale': float(adjustments.get('momentum_scale', 1.0)),
                'preservation_profile': adjustments.get('preservation_profile', 'base-update-family'),
                'objective_profile': adjustments.get('objective_profile', 'none'),
                'witness': dict(adjustments.get('witness', {})),
            })
        self._last_shadow_optimizer_report = {'profile': profile, 'group_reports': group_reports}

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict], shadow_report_getter: Callable[[], dict[str, Any]] | None = None, shadow_acceptance_getter: Callable[[], dict[str, Any]] | None = None):
        super().__init__(param_groups, defaults={})
        self._shadow_report_getter = shadow_report_getter
        self._shadow_acceptance_getter = shadow_acceptance_getter
        self._last_shadow_optimizer_report: dict[str, Any] = {}
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def set_shadow_context(self, shadow_report_getter: Callable[[], dict[str, Any]] | None = None, shadow_acceptance_getter: Callable[[], dict[str, Any]] | None = None) -> None:
        self._shadow_report_getter = shadow_report_getter
        self._shadow_acceptance_getter = shadow_acceptance_getter

    def get_last_shadow_optimizer_report(self) -> dict[str, Any]:
        return dict(self._last_shadow_optimizer_report)

    def _shadow_profile(self) -> dict[str, Any] | None:
        report = self._shadow_report_getter() if self._shadow_report_getter is not None else {}
        acceptance = self._shadow_acceptance_getter() if self._shadow_acceptance_getter is not None else {}
        if not report and not acceptance:
            return None
        return _build_shadow_hyper_profile(report, acceptance)

    def _group_adjustments(self, group: dict, profile: dict[str, Any] | None) -> dict[str, Any]:
        return _shadow_group_adjustments(group, profile)

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int, adjustments: dict[str, Any] | None = None) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        adjustments = adjustments or {'lr_scale': 1.0, 'wd_scale': 1.0}
        effective_lr = float(group['lr']) * float(adjustments.get('lr_scale', 1.0))
        effective_wd = float(group['weight_decay']) * float(adjustments.get('wd_scale', 1.0))
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(effective_lr)
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(effective_wd)
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int, adjustments: dict[str, Any] | None = None) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        adjustments = adjustments or {'lr_scale': 1.0, 'wd_scale': 1.0, 'momentum_scale': 1.0}
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            effective_momentum = _clamp_float(float(group["momentum"]) * float(adjustments.get("momentum_scale", 1.0)), 0.0, 0.9999)
            effective_lr = float(group["lr"]) * float(adjustments.get("lr_scale", 1.0))
            effective_wd = float(group["weight_decay"]) * float(adjustments.get("wd_scale", 1.0))
            self._muon_momentum_t.fill_(effective_momentum)
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(effective_lr * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(effective_wd)
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        profile = self._shadow_profile()
        group_adjustments = [self._group_adjustments(group, profile) for group in self.param_groups]

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        group_reports: list[dict[str, Any]] = []
        for group, info, adjustments in zip(self.param_groups, reduce_infos, group_adjustments):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size, adjustments)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank, adjustments)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
            group_reports.append({
                'role': group.get('role', group.get('kind', 'unknown')),
                'kind': group['kind'],
                'lr_scale': float(adjustments.get('lr_scale', 1.0)),
                'wd_scale': float(adjustments.get('wd_scale', 1.0)),
                'momentum_scale': float(adjustments.get('momentum_scale', 1.0)),
                'preservation_profile': adjustments.get('preservation_profile', 'base-update-family'),
                'objective_profile': adjustments.get('objective_profile', 'none'),
                'witness': dict(adjustments.get('witness', {})),
            })

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
        self._last_shadow_optimizer_report = {'profile': profile, 'group_reports': group_reports}
