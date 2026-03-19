"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.shadowhott import ShadowHoTTOverlay, ShadowState

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # ShadowHoTT overlay: a meta-transformer that reads coarse runtime metadata and emits
    # bounded late-layer modulation plus certified four-valued / bilateral reports.
    shadow_enabled: bool = True
    shadow_layers: int = 4
    shadow_meta_dim: int = 8
    shadow_hidden_dim: int = 128
    shadow_depth: int = 2
    shadow_n_head: int = 4
    shadow_gate_limit: float = 0.20
    shadow_loss_weight: float = 0.05
    shadow_regime_weight: float = 0.02
    shadow_drift_weight: float = 0.01
    shadow_sparsity_weight: float = 0.01
    shadow_provenance_weight: float = 0.005
    shadow_consistency_weight: float = 0.005
    shadow_benchmark_regime_weight: float = 0.50
    shadow_benchmark_drift_weight: float = 0.25
    shadow_benchmark_provenance_weight: float = 0.25
    shadow_benchmark_complexity_weight: float = 0.10
    shadow_adapter_scale: float = 0.12
    shadow_adapter_family_scale: float = 0.08
    shadow_adapter_global_scale: float = 0.05
    shadow_adapter_promotion_rate: float = 0.10
    shadow_adapter_demotion_rate: float = 0.08
    shadow_adapter_reject_penalty: float = 0.12
    shadow_adapter_min_scale: float = 0.05
    shadow_adapter_branch_limit: int = 4
    shadow_candidate_branch_trials: int = 4
    shadow_candidate_branch_perturb_scale: float = 0.10
    shadow_micro_adapter_rank: int = 4
    shadow_branch_learning_rate: float = 0.18
    shadow_inner_loop_steps: int = 2
    shadow_inner_loop_lr: float = 0.12
    shadow_branch_grad_decay: float = 0.92
    shadow_branch_grad_clip: float = 0.25
    shadow_branch_momentum: float = 0.75
    shadow_branch_optimizer_decay: float = 0.98
    shadow_lineage_spawn_threshold: int = 2
    shadow_lineage_prune_threshold: float = -0.20
    shadow_lineage_max_children: int = 2
    shadow_lineage_maturity_generation: int = 1
    shadow_lineage_priority_boost: float = 0.12
    shadow_lineage_search_bonus: int = 1
    shadow_lineage_bandwidth_bonus: int = 1
    shadow_lineage_mature_lr_scale: float = 0.85
    shadow_lineage_newborn_lr_scale: float = 1.10
    shadow_lineage_mature_momentum_bonus: float = 0.08
    shadow_lineage_newborn_momentum_scale: float = 0.92
    shadow_lineage_mature_mutation_scale: float = 0.75
    shadow_lineage_newborn_mutation_scale: float = 1.35
    shadow_lineage_mature_ce_weight: float = 0.85
    shadow_lineage_newborn_ce_weight: float = 1.15
    shadow_lineage_mature_stability_weight: float = 1.25
    shadow_lineage_newborn_stability_weight: float = 0.80
    shadow_lineage_mature_coherence_weight: float = 1.20
    shadow_lineage_newborn_coherence_weight: float = 0.85
    shadow_lineage_spawn_objective_threshold: float = 0.08
    shadow_lineage_prune_objective_threshold: float = -0.05
    shadow_lineage_mature_memory_horizon: int = 8
    shadow_lineage_newborn_memory_horizon: int = 3
    shadow_lineage_mature_replay_decay: float = 0.96
    shadow_lineage_newborn_replay_decay: float = 0.82
    shadow_lineage_fusion_similarity_threshold: float = 0.92
    shadow_lineage_split_variance_threshold: float = 0.18
    shadow_lineage_fusion_priority_bonus: float = 0.10
    shadow_lineage_split_mutation_boost: float = 1.25
    shadow_lineage_fusion_memory_bonus: int = 2
    shadow_lineage_split_memory_scale: float = 0.65
    shadow_lineage_split_replay_decay_scale: float = 0.88


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, attn_scale=None, mlp_scale=None, resid_scale=None):
        attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        mlp_out = self.mlp(norm(x))
        if attn_scale is not None:
            attn_out = attn_out * attn_scale
        if mlp_scale is not None:
            mlp_out = mlp_out * mlp_scale
        if resid_scale is None:
            x = x + attn_out
            x = x + mlp_out
        else:
            x = x + resid_scale * attn_out
            x = x + resid_scale * mlp_out
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.shadowhott = ShadowHoTTOverlay(config) if getattr(config, "shadow_enabled", False) else None
        self.last_loss_components: dict[str, float] = {}
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        self.last_shadow_report = {}

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.shadowhott is not None:
            self.shadowhott.init_weights()

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        shadow_numel = sum(p.numel() for p in self.shadowhott.parameters()) if self.shadowhott is not None else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() + shadow_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        shadow = sum(p.numel() for p in self.shadowhott.parameters()) if self.shadowhott is not None else 0
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + shadow
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'shadowhott': shadow,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        shadow_params = list(self.shadowhott.parameters()) if self.shadowhott is not None else []
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(shadow_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', role='lm_head', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', role='embedding', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', role='value_embedding', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', role='residual_scalar', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', role='x0_scalar', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', role='smear', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', role='shadowhott', params=shadow_params, lr=0.003, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01) if shadow_params else None,
        ]
        param_groups = [g for g in param_groups if g is not None]

        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', role=f'transformer_matrix[{shape}]', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups, shadow_report_getter=self.get_shadow_report, shadow_acceptance_getter=self.get_shadow_acceptance_summary)
        if hasattr(optimizer, 'set_shadow_context'):
            optimizer.set_shadow_context(self.get_shadow_report, self.get_shadow_acceptance_summary)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def reset_shadowhott_state(self):
        if self.shadowhott is not None:
            self.shadowhott.reset_state()
            self.last_shadow_report = {}

    def get_shadow_report(self):
        return self.last_shadow_report

    def get_shadow_acceptance_summary(self):
        if self.shadowhott is None:
            return {}
        return self.shadowhott.get_acceptance_summary()

    def shadow_accept_last(self, score_delta: float, note: str = "", winning_mode: str | None = None, benchmark: dict[str, Any] | None = None):
        if self.shadowhott is None:
            return None
        return self.shadowhott.accept_last_candidate(score_delta=score_delta, note=note, winning_mode=winning_mode, benchmark=benchmark)

    def shadow_reject_last(self, score_delta: float, note: str = "", winning_mode: str | None = None, benchmark: dict[str, Any] | None = None):
        if self.shadowhott is None:
            return None
        return self.shadowhott.reject_last_candidate(score_delta=score_delta, note=note, winning_mode=winning_mode, benchmark=benchmark)

    def _shadow_build_manual_prior(self, attn_coeffs: torch.Tensor, mlp_coeffs: torch.Tensor, resid_coeffs: torch.Tensor, logit_coeffs: torch.Tensor, *, device: torch.device, dtype: torch.dtype, base_prior: dict[str, object], mode: str, branch_key: str) -> dict[str, object]:
        assert self.shadowhott is not None
        limit = float(self.shadowhott.gate_limit)
        attn_delta = self.shadowhott._decode_micro_delta(attn_coeffs, self.shadowhott.attn_adapter_basis).to(device=device, dtype=dtype)
        mlp_delta = self.shadowhott._decode_micro_delta(mlp_coeffs, self.shadowhott.mlp_adapter_basis).to(device=device, dtype=dtype)
        resid_delta = self.shadowhott._decode_micro_delta(resid_coeffs, self.shadowhott.resid_adapter_basis).to(device=device, dtype=dtype)
        logit_delta = self.shadowhott._decode_micro_delta(logit_coeffs, self.shadowhott.logit_adapter_basis.unsqueeze(1)).to(device=device, dtype=dtype).view(() )
        quality = float(base_prior.get('quality', 0.5)) if isinstance(base_prior, dict) else 0.5
        strength = float(base_prior.get('mean_intervention_strength', 0.5)) if isinstance(base_prior, dict) else 0.5
        active_scale = float(base_prior.get('active_scale', 1.0)) if isinstance(base_prior, dict) else 1.0
        return {
            'attn_gates': torch.clamp(1.0 + attn_delta, min=1.0 - limit, max=1.0 + limit),
            'mlp_gates': torch.clamp(1.0 + mlp_delta, min=1.0 - limit, max=1.0 + limit),
            'resid_gates': torch.clamp(1.0 + resid_delta, min=1.0 - limit, max=1.0 + limit),
            'logit_scale': torch.clamp(1.0 + logit_delta, min=torch.tensor(0.8, device=device, dtype=dtype), max=torch.tensor(1.2, device=device, dtype=dtype)),
            'template_count': int(base_prior.get('template_count', 1)) if isinstance(base_prior, dict) else 1,
            'mode_counts': dict(base_prior.get('mode_counts', {})) if isinstance(base_prior, dict) else {},
            'preferred_mode': mode,
            'branch_key': branch_key,
            'quality': quality,
            'mean_intervention_strength': strength,
            'active_scale': active_scale,
            'attn_micro_coeffs': attn_coeffs,
            'mlp_micro_coeffs': mlp_coeffs,
            'resid_micro_coeffs': resid_coeffs,
            'logit_micro_coeffs': logit_coeffs,
        }

    def _shadow_zero_optimizer_state(self, device: torch.device) -> dict[str, torch.Tensor | int | float]:
        assert self.shadowhott is not None
        rank = int(self.shadowhott.micro_adapter_rank)
        return {
            'attn_velocity': torch.zeros(rank, device=device),
            'mlp_velocity': torch.zeros(rank, device=device),
            'resid_velocity': torch.zeros(rank, device=device),
            'logit_velocity': torch.zeros(rank, device=device),
            'steps': 0,
            'last_lr': 0.0,
            'last_momentum': 0.0,
        }

    def _shadow_prepare_optimizer_state(self, optimizer_state: dict[str, object] | None, device: torch.device) -> dict[str, torch.Tensor | int | float]:
        state = self._shadow_zero_optimizer_state(device)
        if optimizer_state is None:
            return state
        for key in ['attn_velocity', 'mlp_velocity', 'resid_velocity', 'logit_velocity']:
            val = optimizer_state.get(key) if isinstance(optimizer_state, dict) else None
            if val is not None:
                state[key] = torch.as_tensor(val, device=device).detach().float().clone()
        state['steps'] = int(optimizer_state.get('steps', 0)) if isinstance(optimizer_state, dict) else 0
        state['last_lr'] = float(optimizer_state.get('last_lr', 0.0)) if isinstance(optimizer_state, dict) else 0.0
        state['last_momentum'] = float(optimizer_state.get('last_momentum', 0.0)) if isinstance(optimizer_state, dict) else 0.0
        return state


    def _shadow_lineage_objective_weights(self, lineage_profile: dict[str, object] | None = None) -> dict[str, float]:
        profile = lineage_profile or {}
        mature = bool(profile.get('mature', False))
        if mature:
            ce_scale = float(self.config.shadow_lineage_mature_ce_weight)
            stability_scale = float(self.config.shadow_lineage_mature_stability_weight)
            coherence_scale = float(self.config.shadow_lineage_mature_coherence_weight)
        else:
            ce_scale = float(self.config.shadow_lineage_newborn_ce_weight)
            stability_scale = float(self.config.shadow_lineage_newborn_stability_weight)
            coherence_scale = float(self.config.shadow_lineage_newborn_coherence_weight)
        return {
            'ce': ce_scale,
            'regime': float(self.config.shadow_benchmark_regime_weight) * ce_scale,
            'drift': float(self.config.shadow_benchmark_drift_weight) * stability_scale,
            'provenance': float(self.config.shadow_benchmark_provenance_weight) * coherence_scale,
            'consistency': float(self.config.shadow_consistency_weight) * coherence_scale,
            'sparsity': float(self.config.shadow_sparsity_weight) * stability_scale,
            'complexity': float(self.config.shadow_benchmark_complexity_weight) * stability_scale,
        }

    def _shadow_objective_score(self, baseline_components: dict[str, float], candidate_components: dict[str, float], lineage_profile: dict[str, object] | None = None) -> tuple[float, dict[str, float]]:
        weights = self._shadow_lineage_objective_weights(lineage_profile)
        base_ce = float(baseline_components.get('ce_loss', baseline_components.get('total_loss', 0.0)))
        cand_ce = float(candidate_components.get('ce_loss', candidate_components.get('total_loss', 0.0)))
        gains = {
            'ce_gain': base_ce - cand_ce,
            'regime_gain': float(baseline_components.get('regime_loss', 0.0)) - float(candidate_components.get('regime_loss', 0.0)),
            'drift_gain': float(baseline_components.get('drift_loss', 0.0)) - float(candidate_components.get('drift_loss', 0.0)),
            'provenance_gain': float(baseline_components.get('provenance_loss', 0.0)) - float(candidate_components.get('provenance_loss', 0.0)),
            'consistency_gain': float(baseline_components.get('consistency_loss', 0.0)) - float(candidate_components.get('consistency_loss', 0.0)),
            'sparsity_gain': float(baseline_components.get('sparsity_loss', 0.0)) - float(candidate_components.get('sparsity_loss', 0.0)),
            'complexity_penalty': max(0.0, float(candidate_components.get('control_regularization', 0.0)) - float(baseline_components.get('control_regularization', 0.0))),
        }
        score = (
            weights['ce'] * gains['ce_gain']
            + weights['regime'] * gains['regime_gain']
            + weights['drift'] * gains['drift_gain']
            + weights['provenance'] * gains['provenance_gain']
            + weights['consistency'] * gains['consistency_gain']
            + weights['sparsity'] * gains['sparsity_gain']
            - weights['complexity'] * gains['complexity_penalty']
        )
        return float(score), {'weights': weights, 'gains': gains}

    def _shadow_inner_loop_candidate(self, idx, targets, candidate_spec: dict[str, object], *, base_context: str) -> dict[str, object]:
        assert self.shadowhott is not None
        device = idx.device
        dtype = self.lm_head.weight.dtype
        base_prior = dict(candidate_spec.get('prior', {}))
        if not base_prior:
            return {'mode': str(candidate_spec.get('mode', 'candidate_branch:empty')), 'loss': float('inf'), 'components': {}, 'branch_key': str(candidate_spec.get('branch_key', 'none')), 'blend': float(candidate_spec.get('blend', 0.0)), 'steps_run': 0, 'replay_priority': float(candidate_spec.get('replay_priority', 0.0)), 'lineage_bandwidth_bonus': int(candidate_spec.get('lineage_bandwidth_bonus', 0))}
        attn0 = self.shadowhott._project_micro_coeffs(torch.as_tensor(base_prior['attn_gates']).detach().float() - 1.0, self.shadowhott.attn_adapter_basis).to(device=device)
        mlp0 = self.shadowhott._project_micro_coeffs(torch.as_tensor(base_prior['mlp_gates']).detach().float() - 1.0, self.shadowhott.mlp_adapter_basis).to(device=device)
        resid0 = self.shadowhott._project_micro_coeffs(torch.as_tensor(base_prior['resid_gates']).detach().float() - 1.0, self.shadowhott.resid_adapter_basis).to(device=device)
        logit_scale = base_prior.get('logit_scale', 1.0)
        if torch.is_tensor(logit_scale):
            logit_scale_val = float(logit_scale.detach().float().item())
        else:
            logit_scale_val = float(logit_scale)
        logit0 = self.shadowhott._project_micro_coeffs(torch.tensor([logit_scale_val - 1.0], device=device), self.shadowhott.logit_adapter_basis.unsqueeze(1)).to(device=device)
        trajectory_bias = candidate_spec.get('trajectory_bias', {}) if isinstance(candidate_spec, dict) else {}
        if isinstance(trajectory_bias, dict):
            attn0 = attn0 + torch.as_tensor(trajectory_bias.get('attn_micro_coeffs', torch.zeros_like(attn0)), device=device).detach().float()
            mlp0 = mlp0 + torch.as_tensor(trajectory_bias.get('mlp_micro_coeffs', torch.zeros_like(mlp0)), device=device).detach().float()
            resid0 = resid0 + torch.as_tensor(trajectory_bias.get('resid_micro_coeffs', torch.zeros_like(resid0)), device=device).detach().float()
            logit0 = logit0 + torch.as_tensor(trajectory_bias.get('logit_micro_coeffs', torch.zeros_like(logit0)), device=device).detach().float()
        coeffs = [attn0.detach().clone().requires_grad_(True), mlp0.detach().clone().requires_grad_(True), resid0.detach().clone().requires_grad_(True), logit0.detach().clone().requires_grad_(True)]
        optimizer_state = self._shadow_prepare_optimizer_state(candidate_spec.get('optimizer_state') if isinstance(candidate_spec, dict) else None, device)
        steps_run = 0
        train_state = self.training
        self.eval()
        with torch.enable_grad():
            inner_steps = max(0, int(self.config.shadow_inner_loop_steps) + int(candidate_spec.get('inner_loop_steps_bonus', 0)))
            for _ in range(inner_steps):
                manual_prior = self._shadow_build_manual_prior(coeffs[0], coeffs[1], coeffs[2], coeffs[3], device=device, dtype=dtype, base_prior=base_prior, mode=str(candidate_spec.get('mode', 'candidate_branch:inner')), branch_key=str(candidate_spec.get('branch_key', 'none')))
                loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=False, shadow_apply_persistent_prior=False, shadow_apply_episode_gate_prior=False, shadow_add_regularization=True, shadow_context_tag=base_context, shadow_manual_prior=manual_prior, shadow_manual_blend=float(candidate_spec.get('blend', 0.0)), shadow_manual_mode=str(candidate_spec.get('mode', 'candidate_branch:inner')))
                grads = torch.autograd.grad(loss, coeffs, allow_unused=True)
                new_coeffs = []
                lr = float(self.config.shadow_inner_loop_lr) * float(candidate_spec.get('inner_loop_lr_scale', 1.0))
                momentum = float(candidate_spec.get('inner_loop_momentum', float(self.config.shadow_branch_momentum)))
                decay = float(self.config.shadow_branch_optimizer_decay)
                for name, c, g in zip(['attn', 'mlp', 'resid', 'logit'], coeffs, grads):
                    vel_key = f'{name}_velocity'
                    vel = torch.as_tensor(optimizer_state.get(vel_key), device=device).detach().float().clone()
                    if g is None:
                        new_coeffs.append(c.detach().clone().requires_grad_(True))
                        optimizer_state[vel_key] = decay * vel
                    else:
                        new_vel = momentum * decay * vel - lr * g.detach().float()
                        upd = c + new_vel
                        new_coeffs.append(upd.detach().clone().requires_grad_(True))
                        optimizer_state[vel_key] = new_vel.detach()
                optimizer_state['steps'] = int(optimizer_state.get('steps', 0)) + 1
                optimizer_state['last_lr'] = lr
                optimizer_state['last_momentum'] = momentum
                coeffs = new_coeffs
                steps_run += 1
        final_prior = self._shadow_build_manual_prior(coeffs[0].detach(), coeffs[1].detach(), coeffs[2].detach(), coeffs[3].detach(), device=device, dtype=dtype, base_prior=base_prior, mode=str(candidate_spec.get('mode', 'candidate_branch:inner')), branch_key=str(candidate_spec.get('branch_key', 'none')))
        with torch.no_grad():
            final_loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=False, shadow_apply_persistent_prior=False, shadow_apply_episode_gate_prior=False, shadow_add_regularization=True, shadow_context_tag=base_context, shadow_manual_prior=final_prior, shadow_manual_blend=float(candidate_spec.get('blend', 0.0)), shadow_manual_mode=str(candidate_spec.get('mode', 'candidate_branch:inner')))
            final_components = dict(getattr(self, 'last_loss_components', {}))
        if train_state:
            self.train()
        return {
            'mode': str(candidate_spec.get('mode', 'candidate_branch:inner')),
            'branch_key': str(candidate_spec.get('branch_key', 'none')),
            'blend': float(candidate_spec.get('blend', 0.0)),
            'loss': float(final_loss.detach().float().item()),
            'components': final_components,
            'steps_run': steps_run,
            'manual_prior': {k: (v.detach() if torch.is_tensor(v) else v) for k, v in final_prior.items()},
            'inner_loop_coeff_norm': float(sum(torch.norm(c.detach()).item() for c in coeffs)),
            'optimizer_state': {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in optimizer_state.items()},
            'optimizer_steps': int(optimizer_state.get('steps', 0)),
            'optimizer_last_lr': float(optimizer_state.get('last_lr', 0.0)),
            'optimizer_last_momentum': float(optimizer_state.get('last_momentum', float(candidate_spec.get('inner_loop_momentum', self.config.shadow_branch_momentum)))),
            'inner_loop_lr_scale': float(candidate_spec.get('inner_loop_lr_scale', 1.0)),
            'lineage_profile': dict(candidate_spec.get('lineage_profile', {})) if isinstance(candidate_spec, dict) else {},
            'trajectory_conditioned': bool(candidate_spec.get('trajectory_conditioned', False)) if isinstance(candidate_spec, dict) else False,
            'lineage_spawned': bool(candidate_spec.get('lineage_spawned', False)) if isinstance(candidate_spec, dict) else False,
        }

    def shadow_benchmark_batch(self, idx, targets, accept_threshold: float = 0.0, reject_threshold: float = 0.0, note_prefix: str = "auto", shadow_context_tag: str | None = None):
        if self.shadowhott is None:
            return {'enabled': False, 'decision': 'disabled'}
        was_training = self.training
        self.eval()
        base_context = shadow_context_tag or 'generic'
        with torch.no_grad():
            baseline_loss = self(idx, targets, shadow_enabled_override=False, shadow_record_candidate=False, shadow_add_regularization=False, shadow_context_tag=base_context)
            baseline_components = dict(getattr(self, 'last_loss_components', {}))
            shadow_live_loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=False, shadow_apply_persistent_prior=True, shadow_apply_episode_gate_prior=False, shadow_add_regularization=True, shadow_context_tag=base_context)
            live_components = dict(getattr(self, 'last_loss_components', {}))
            self.shadowhott.begin_episode(base_context)
            pre_signature = self.shadowhott.current_context_signature(base_context, self.shadowhott.get_state())
            prior, blend, source = self.shadowhott._lookup_signature_prior(pre_signature)
            if prior is not None and blend > 0.0:
                self.shadowhott._episode_active_signature = pre_signature
                self.shadowhott._episode_active_prior = prior
                self.shadowhott._episode_active_prior_source = f"episode_benchmark:{source}"
                self.shadowhott._episode_active_prior_blend = float(min(0.55, blend + 0.10))
            shadow_gate_loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=False, shadow_apply_persistent_prior=True, shadow_apply_episode_gate_prior=True, shadow_add_regularization=True, shadow_context_tag=base_context)
            gate_components = dict(getattr(self, 'last_loss_components', {}))
            self.shadowhott.end_episode()
        base_val = float(baseline_loss.detach().float().item())
        live_val = float(shadow_live_loss.detach().float().item())
        gate_val = float(shadow_gate_loss.detach().float().item())
        gate_vs_live_delta = live_val - gate_val
        if gate_val < live_val:
            chosen_mode = 'episode_gate'
            chosen_kwargs = dict(shadow_apply_episode_gate_prior=True)
        else:
            chosen_mode = 'live_only'
            chosen_kwargs = dict(shadow_apply_episode_gate_prior=False)
        with torch.no_grad():
            if chosen_mode == 'episode_gate':
                self.shadowhott.begin_episode(base_context)
                prior, blend, source = self.shadowhott._lookup_signature_prior(pre_signature)
                if prior is not None and blend > 0.0:
                    self.shadowhott._episode_active_signature = pre_signature
                    self.shadowhott._episode_active_prior = prior
                    self.shadowhott._episode_active_prior_source = f"episode_benchmark:{source}"
                    self.shadowhott._episode_active_prior_blend = float(min(0.55, blend + 0.10))
            chosen_loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=True, shadow_apply_persistent_prior=True, shadow_add_regularization=True, shadow_context_tag=base_context, **chosen_kwargs)
            chosen_components = dict(getattr(self, 'last_loss_components', {}))
            if chosen_mode == 'episode_gate':
                self.shadowhott.end_episode()
        selected_result = {
            'mode': chosen_mode,
            'branch_key': 'live_control',
            'blend': 0.0,
            'loss': float(chosen_loss.detach().float().item()),
            'components': chosen_components,
            'steps_run': 0,
            'manual_prior': None,
            'inner_loop_coeff_norm': 0.0,
            'candidate_type': 'live',
            'lineage_profile': {'mature': False},
        }
        live_objective_score, live_objective_meta = self._shadow_objective_score(baseline_components, selected_result['components'], selected_result.get('lineage_profile'))
        selected_result['objective_score'] = float(live_objective_score)
        selected_result['objective_weights'] = dict(live_objective_meta['weights'])
        selected_result['objective_gains'] = dict(live_objective_meta['gains'])
        candidate_results: list[dict[str, object]] = []
        if self.shadowhott is not None and self.shadowhott._last_candidate is not None:
            for spec in self.shadowhott.candidate_branch_specs(self.shadowhott._last_candidate, context_signature=str(self.shadowhott._last_candidate.get('context_signature', base_context))):
                result = self._shadow_inner_loop_candidate(idx, targets, spec, base_context=base_context)
                result['candidate_type'] = 'inner_loop_branch'
                objective_score, objective_meta = self._shadow_objective_score(baseline_components, result.get('components', {}), result.get('lineage_profile'))
                result['objective_score'] = float(objective_score)
                result['objective_weights'] = dict(objective_meta['weights'])
                result['objective_gains'] = dict(objective_meta['gains'])
                candidate_results.append(result)
            if candidate_results:
                best_candidate = max(candidate_results, key=lambda r: float(r.get('objective_score', float('-inf'))))
                if float(best_candidate.get('objective_score', float('-inf'))) > float(selected_result.get('objective_score', float('-inf'))):
                    selected_result = best_candidate
                    with torch.no_grad():
                        final_loss = self(idx, targets, shadow_enabled_override=True, shadow_record_candidate=True, shadow_apply_persistent_prior=False, shadow_apply_episode_gate_prior=False, shadow_add_regularization=True, shadow_context_tag=base_context, shadow_manual_prior=best_candidate['manual_prior'], shadow_manual_blend=float(best_candidate['blend']), shadow_manual_mode=str(best_candidate['mode']))
                        chosen_components = dict(getattr(self, 'last_loss_components', {}))
                    selected_result['loss'] = float(final_loss.detach().float().item())
                    selected_result['components'] = chosen_components
                    objective_score, objective_meta = self._shadow_objective_score(baseline_components, chosen_components, selected_result.get('lineage_profile'))
                    selected_result['objective_score'] = float(objective_score)
                    selected_result['objective_weights'] = dict(objective_meta['weights'])
                    selected_result['objective_gains'] = dict(objective_meta['gains'])
        shadow_val = float(selected_result['loss'])
        score_delta = base_val - shadow_val
        composite_delta = float(selected_result.get('objective_score', 0.0))
        if self.shadowhott is not None and str(selected_result.get('candidate_type', 'live')) == 'inner_loop_branch':
            self.shadowhott.update_branch_optimizer_state(
                context_signature=str(self.shadowhott._last_candidate.get('context_signature', base_context)) if self.shadowhott._last_candidate is not None else base_context,
                branch_key=str(selected_result.get('branch_key', 'none')),
                optimizer_state=selected_result.get('optimizer_state'),
                score_delta=score_delta,
            )
        benchmark_summary = {
            'baseline_loss': base_val,
            'shadow_loss': shadow_val,
            'shadow_live_loss': live_val,
            'shadow_gate_loss': gate_val,
            'gate_vs_live_delta': gate_vs_live_delta,
            'selected_shadow_mode': str(selected_result['mode']),
            'selected_branch_key': str(selected_result.get('branch_key', 'live_control')),
            'selected_candidate_type': str(selected_result.get('candidate_type', 'live')),
            'selected_inner_loop_steps': int(selected_result.get('steps_run', 0)),
            'selected_optimizer_steps': int(selected_result.get('optimizer_steps', 0)),
            'selected_optimizer_last_lr': float(selected_result.get('optimizer_last_lr', 0.0)),
            'selected_objective_score': float(selected_result.get('objective_score', 0.0)),
            'selected_objective_weights': dict(selected_result.get('objective_weights', {})),
            'selected_objective_gains': dict(selected_result.get('objective_gains', {})),
            'candidate_branch_trials': len(candidate_results),
            'candidate_branch_results': [
                {
                    'mode': str(r['mode']),
                    'branch_key': str(r.get('branch_key', 'none')),
                    'loss': float(r['loss']),
                    'steps_run': int(r.get('steps_run', 0)),
                    'blend': float(r.get('blend', 0.0)),
                    'inner_loop_coeff_norm': float(r.get('inner_loop_coeff_norm', 0.0)),
                    'optimizer_steps': int(r.get('optimizer_steps', 0)),
                    'optimizer_last_lr': float(r.get('optimizer_last_lr', 0.0)),
                    'trajectory_conditioned': bool(r.get('trajectory_conditioned', False)),
                    'lineage_spawned': bool(r.get('lineage_spawned', False)),
                    'replay_priority': float(r.get('replay_priority', 0.0)),
                    'lineage_bandwidth_bonus': int(r.get('lineage_bandwidth_bonus', 0)),
                    'objective_score': float(r.get('objective_score', 0.0)),
                    'objective_weights': dict(r.get('objective_weights', {})),
                    'objective_gains': dict(r.get('objective_gains', {})),
                }
                for r in candidate_results
            ],
            'shadow_context_tag': base_context,
            'baseline_components': baseline_components,
            'shadow_live_components': live_components,
            'shadow_gate_components': gate_components,
            'shadow_selected_components': selected_result['components'],
            'composite_score_delta': composite_delta,
            'lineage_objective_weighting': True,
        }
        if self.shadowhott is not None and self.shadowhott._last_candidate is not None:
            self.shadowhott._last_candidate['selected_shadow_mode'] = str(selected_result['mode'])
            if 'parent_branch_key' in selected_result:
                self.shadowhott._last_candidate['parent_branch_key'] = selected_result.get('parent_branch_key')
        if 'fused_from' in selected_result:
            self.shadowhott._last_candidate['fused_from'] = list(selected_result.get('fused_from', []))
        if 'split_from' in selected_result:
            self.shadowhott._last_candidate['split_from'] = selected_result.get('split_from')
        if 'memory_parent_branches' in selected_result:
            self.shadowhott._last_candidate['memory_parent_branches'] = list(selected_result.get('memory_parent_branches', []))
        if 'memory_horizon' in selected_result:
            self.shadowhott._last_candidate['memory_horizon'] = int(selected_result.get('memory_horizon', 1))
        if 'replay_decay' in selected_result:
            self.shadowhott._last_candidate['replay_decay'] = float(selected_result.get('replay_decay', 1.0))
        self.shadowhott._last_candidate['benchmark'] = dict(benchmark_summary)
        decision = 'defer'
        record = None
        if score_delta > accept_threshold:
            decision = 'accept'
            record = self.shadow_accept_last(score_delta=score_delta, note=f"{note_prefix}: baseline={base_val:.6f} shadow={shadow_val:.6f} composite={composite_delta:.6f} live={live_val:.6f} gate={gate_val:.6f} mode={selected_result['mode']} context={base_context}", winning_mode=str(selected_result['mode']), benchmark=benchmark_summary)
        elif score_delta < -reject_threshold:
            decision = 'reject'
            record = self.shadow_reject_last(score_delta=score_delta, note=f"{note_prefix}: baseline={base_val:.6f} shadow={shadow_val:.6f} composite={composite_delta:.6f} live={live_val:.6f} gate={gate_val:.6f} mode={selected_result['mode']} context={base_context}", winning_mode=str(selected_result['mode']), benchmark=benchmark_summary)
        if was_training:
            self.train()
        return {
            'enabled': True,
            'baseline_loss': base_val,
            'shadow_loss': shadow_val,
            'shadow_live_loss': live_val,
            'shadow_gate_loss': gate_val,
            'gate_vs_live_delta': gate_vs_live_delta,
            'selected_shadow_mode': str(selected_result['mode']),
            'selected_branch_key': str(selected_result.get('branch_key', 'live_control')),
            'selected_candidate_type': str(selected_result.get('candidate_type', 'live')),
            'selected_inner_loop_steps': int(selected_result.get('steps_run', 0)),
            'selected_optimizer_steps': int(selected_result.get('optimizer_steps', 0)),
            'selected_optimizer_last_lr': float(selected_result.get('optimizer_last_lr', 0.0)),
            'selected_objective_score': float(selected_result.get('objective_score', 0.0)),
            'selected_objective_weights': dict(selected_result.get('objective_weights', {})),
            'selected_objective_gains': dict(selected_result.get('objective_gains', {})),
            'candidate_branch_trials': len(candidate_results),
            'score_delta': score_delta,
            'composite_score_delta': composite_delta,
            'decision': decision,
            'record': record,
            'accept_threshold': float(accept_threshold),
            'reject_threshold': float(reject_threshold),
            'shadow_report': self.get_shadow_report(),
            'acceptance_summary': self.get_shadow_acceptance_summary(),
            'shadow_context_tag': base_context,
            'baseline_components': baseline_components,
            'shadow_selected_components': selected_result['components'],
            'candidate_branch_results': benchmark_summary['candidate_branch_results'],
        }

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', shadow_enabled_override=None, shadow_record_candidate=True, shadow_apply_persistent_prior=True, shadow_apply_episode_gate_prior=True, shadow_add_regularization=True, shadow_context_tag: str | None = None, shadow_episode_step: int | None = None, shadow_manual_prior: dict[str, object] | None = None, shadow_manual_blend: float = 0.0, shadow_manual_mode: str | None = None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        shadow_control = None
        shadow_prev_state = None
        shadow_context_signature = 'generic|truth'
        episode_gate_reroute_info = {'applied': False, 'source': 'none', 'blend': 0.0, 'active_signature': None, 'preferred_mode': 'live_only', 'selected_mode': 'live_only'}
        shadow_active = self.shadowhott is not None if shadow_enabled_override is None else bool(shadow_enabled_override and self.shadowhott is not None)
        if shadow_active:
            if targets is not None and self.training:
                shadow_prev_state = ShadowState()
            else:
                shadow_prev_state = self.shadowhott.get_state()
            shadow_tokens = self.shadowhott.build_pre_tokens(idx, x, shadow_prev_state, self.window_sizes[-1][0], in_training=targets is not None)
            shadow_context_signature = self.shadowhott.current_context_signature(shadow_context_tag, shadow_prev_state)
            shadow_control = self.shadowhott.meta(shadow_tokens)
            if shadow_manual_prior is not None and float(shadow_manual_blend) > 0.0:
                shadow_control = self.shadowhott._blend_control_with_prior(shadow_control, shadow_manual_prior, float(shadow_manual_blend))
                episode_gate_reroute_info['preferred_mode'] = shadow_manual_mode or 'manual_prior'
                episode_gate_reroute_info['selected_mode'] = shadow_manual_mode or 'manual_prior'
                episode_gate_reroute_info['manual_prior'] = True
            elif shadow_apply_persistent_prior:
                preferred_mode = self.shadowhott.get_preferred_mode(shadow_context_signature)
                if shadow_apply_episode_gate_prior and preferred_mode == 'episode_gate':
                    shadow_control, episode_gate_reroute_info = self.shadowhott.apply_preferred_mode_prior(shadow_control, context_signature=shadow_context_signature)
                else:
                    shadow_control = self.shadowhott.apply_persistent_prior(shadow_control, context_signature=shadow_context_signature)
                    episode_gate_reroute_info['preferred_mode'] = preferred_mode
                    if shadow_apply_episode_gate_prior:
                        shadow_control, episode_gate_reroute_info = self.shadowhott.apply_episode_active_gate_prior(shadow_control, context_signature=shadow_context_signature)
                        episode_gate_reroute_info['preferred_mode'] = preferred_mode
                        if episode_gate_reroute_info.get('applied', False):
                            episode_gate_reroute_info['selected_mode'] = 'episode_gate'
                        else:
                            episode_gate_reroute_info.setdefault('selected_mode', 'live_only')
                    else:
                        episode_gate_reroute_info['selected_mode'] = 'live_only'

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        shadow_start = max(0, n_layer - (self.shadowhott.shadow_layers if self.shadowhott is not None else 0))
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            attn_scale = mlp_scale = resid_scale = None
            if shadow_control is not None and i >= shadow_start:
                gate_idx = i - shadow_start
                attn_scale = shadow_control.attn_gates[:, gate_idx].to(dtype=x.dtype)[:, None, None]
                mlp_scale = shadow_control.mlp_gates[:, gate_idx].to(dtype=x.dtype)[:, None, None]
                resid_scale = shadow_control.resid_gates[:, gate_idx].to(dtype=x.dtype)[:, None, None]
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, attn_scale=attn_scale, mlp_scale=mlp_scale, resid_scale=resid_scale)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        if shadow_control is not None:
            logits = self.shadowhott.apply_logits(logits, shadow_control)
            logits, live_reroute_info = self.shadowhott.apply_live_episode_reroute(
                logits,
                targets,
                shadow_prev_state,
                shadow_control,
                pre_context_signature=shadow_context_signature,
                context_tag=shadow_context_tag,
            )
            self.last_shadow_report = self.shadowhott.update_and_report(
                logits,
                targets,
                shadow_prev_state,
                shadow_control,
                record_candidate=shadow_record_candidate,
                context_signature=shadow_context_signature,
                context_tag=shadow_context_tag,
                episode_step=shadow_episode_step,
                sequence_len=T0 + T,
                live_reroute_info={**live_reroute_info, 'episode_gate_reroute': episode_gate_reroute_info},
            )
        else:
            self.last_shadow_report = {}

        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            loss = ce_loss
            loss_components = {
                'ce_loss': float(ce_loss.detach().float().item()),
                'control_reg': 0.0,
                'regime_loss': 0.0,
                'drift_loss': 0.0,
                'sparsity_loss': 0.0,
                'provenance_loss': 0.0,
                'consistency_loss': 0.0,
                'total_loss': float(ce_loss.detach().float().item()),
            }
            if shadow_control is not None and shadow_add_regularization:
                control_reg = self.shadowhott.control_regularization(shadow_control)
                shadow_losses = self.shadowhott.compute_shadow_losses(
                    logits,
                    targets,
                    shadow_prev_state,
                    shadow_control,
                    context_signature=shadow_context_signature,
                    report=self.last_shadow_report,
                )
                loss = loss + self.config.shadow_loss_weight * control_reg
                loss = loss + self.config.shadow_regime_weight * shadow_losses['regime']
                loss = loss + self.config.shadow_drift_weight * shadow_losses['drift']
                loss = loss + self.config.shadow_sparsity_weight * shadow_losses['sparsity']
                loss = loss + self.config.shadow_provenance_weight * shadow_losses['provenance']
                loss = loss + self.config.shadow_consistency_weight * shadow_losses['consistency']
                loss_components.update({
                    'control_reg': float(control_reg.detach().float().item()),
                    'regime_loss': float(shadow_losses['regime'].detach().float().item()),
                    'drift_loss': float(shadow_losses['drift'].detach().float().item()),
                    'sparsity_loss': float(shadow_losses['sparsity'].detach().float().item()),
                    'provenance_loss': float(shadow_losses['provenance'].detach().float().item()),
                    'consistency_loss': float(shadow_losses['consistency'].detach().float().item()),
                })
            loss_components['total_loss'] = float(loss.detach().float().item())
            self.last_loss_components = loss_components
            return loss
        else:
            self.last_loss_components = {}
            return logits

    @torch.inference_mode()
    def generate_episode(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42, shadow_context_tag: str | None = None):
        assert isinstance(tokens, list)
        device = self.get_device()
        if self.shadowhott is not None:
            self.reset_shadowhott_state()
            self.shadowhott.begin_episode(shadow_context_tag)
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        class _ShadowEpisodeKVCache:
            def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
                self.batch_size = batch_size
                self.max_seq_len = seq_len
                self.n_layers = num_layers
                self.n_heads = num_heads
                self.head_dim = head_dim
                self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
                self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
                self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
                self.prev_embedding = None

            def get_pos(self):
                return self.cache_seqlens[0].item()

            def get_layer_cache(self, layer_idx):
                return self.k_cache[layer_idx], self.v_cache[layer_idx]

            def advance(self, num_tokens):
                self.cache_seqlens += num_tokens

        kv_cache = _ShadowEpisodeKVCache(
            batch_size=1,
            num_heads=self.config.n_kv_head,
            seq_len=self.config.sequence_len,
            head_dim=self.config.n_embd // self.config.n_head,
            num_layers=self.config.n_layer,
            device=device,
            dtype=COMPUTE_DTYPE,
        )
        # Prefill prompt into cache and initialize live shadow routing on the prompt context.
        _ = self.forward(ids, kv_cache=kv_cache, shadow_context_tag=shadow_context_tag, shadow_record_candidate=False, shadow_apply_persistent_prior=True, shadow_episode_step=0)
        generated = []
        current = ids[:, -1:]
        for step in range(max_tokens):
            logits = self.forward(current, kv_cache=kv_cache, shadow_context_tag=shadow_context_tag, shadow_record_candidate=False, shadow_apply_persistent_prior=True, shadow_episode_step=step + 1)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.clone()
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            token = int(next_ids.item())
            generated.append(token)
            current = next_ids
        episode = self.shadowhott.end_episode() if self.shadowhott is not None else {'steps': 0, 'trace': []}
        return {
            'prompt_tokens': list(tokens),
            'generated_tokens': generated,
            'shadow_episode': episode,
            'shadow_report': self.get_shadow_report(),
        }

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        if self.shadowhott is not None:
            self.reset_shadowhott_state()
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
