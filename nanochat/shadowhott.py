from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, field
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean() if x.numel() else torch.zeros((), device=x.device, dtype=torch.float32)


@dataclass
class ShadowState:
    truth_mass: float = 0.0
    falsity_mass: float = 0.0
    both_mass: float = 0.0
    neither_mass: float = 1.0
    provenance_coherence: float = 1.0
    mutation_risk: float = 0.0
    entropy: float = 0.0
    margin: float = 0.0
    step: int = 0

    def as_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor([
            self.truth_mass,
            self.falsity_mass,
            self.both_mass,
            self.neither_mass,
            self.provenance_coherence,
            self.mutation_risk,
            self.entropy,
            self.margin,
        ], device=device, dtype=dtype)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class ShadowControl:
    attn_gates: torch.Tensor
    mlp_gates: torch.Tensor
    resid_gates: torch.Tensor
    logit_scale: torch.Tensor
    intervention_strength: torch.Tensor
    persist_score: torch.Tensor

    def summary(self) -> dict[str, float]:
        return {
            'attn_mean': float(self.attn_gates.detach().float().mean().item()),
            'mlp_mean': float(self.mlp_gates.detach().float().mean().item()),
            'resid_mean': float(self.resid_gates.detach().float().mean().item()),
            'logit_scale_mean': float(self.logit_scale.detach().float().mean().item()),
            'intervention_strength': float(self.intervention_strength.detach().float().mean().item()),
            'persist_score': float(self.persist_score.detach().float().mean().item()),
        }




@dataclass
class ShadowAcceptanceRecord:
    candidate_id: int
    action: str
    score_delta: float
    note: str = ""
    context_signature: str = "generic|truth"
    control_summary: dict[str, float] = field(default_factory=dict)
    four_value_histogram: dict[str, float] = field(default_factory=dict)


class ShadowMetaTransformer(nn.Module):
    def __init__(self, meta_dim: int, hidden_dim: int, depth: int, n_head: int, shadow_layers: int, gate_limit: float):
        super().__init__()
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.shadow_layers = shadow_layers
        self.gate_limit = gate_limit
        self.token_projs = nn.ModuleList([nn.Linear(meta_dim, hidden_dim, bias=False) for _ in range(4)])
        self.type_embed = nn.Embedding(4, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        out_dim = shadow_layers * 3 + 3
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def init_weights(self):
        for proj in self.token_projs:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)
        for mod in self.head:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, tokens: torch.Tensor) -> ShadowControl:
        # tokens: (B, 4, meta_dim)
        pieces = []
        for i, proj in enumerate(self.token_projs):
            z = proj(tokens[:, i]) + self.type_embed.weight[i]
            pieces.append(z.unsqueeze(1))
        z = torch.cat(pieces, dim=1)
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        raw = self.head(pooled)
        k = self.shadow_layers
        attn_raw = raw[:, 0:k]
        mlp_raw = raw[:, k:2*k]
        resid_raw = raw[:, 2*k:3*k]
        logit_raw = raw[:, 3*k]
        intervene_raw = raw[:, 3*k + 1]
        persist_raw = raw[:, 3*k + 2]
        limit = self.gate_limit
        attn = 1.0 + limit * torch.tanh(attn_raw)
        mlp = 1.0 + limit * torch.tanh(mlp_raw)
        resid = 1.0 + limit * torch.tanh(resid_raw)
        logit_scale = 1.0 + 0.20 * torch.tanh(logit_raw)
        intervention_strength = torch.sigmoid(intervene_raw)
        persist_score = torch.sigmoid(persist_raw)
        return ShadowControl(
            attn_gates=attn,
            mlp_gates=mlp,
            resid_gates=resid,
            logit_scale=logit_scale,
            intervention_strength=intervention_strength,
            persist_score=persist_score,
        )


class ShadowHoTTOverlay(nn.Module):
    """
    A bounded ShadowHoTT overlay for a standard transformer.

    It keeps the base transformer architecture intact and adds a second-layer meta-transformer
    that reads coarse metadata, maintains a four-valued / bilateral shadow state, emits bounded
    late-layer intervention gates, and produces invariant/certification reports.
    """
    def __init__(self, config):
        super().__init__()
        self.enabled = getattr(config, 'shadow_enabled', True)
        self.shadow_layers = min(getattr(config, 'shadow_layers', 4), config.n_layer)
        self.meta_dim = getattr(config, 'shadow_meta_dim', 8)
        self.hidden_dim = getattr(config, 'shadow_hidden_dim', max(64, config.n_embd // 4))
        self.depth = getattr(config, 'shadow_depth', 2)
        self.n_head = getattr(config, 'shadow_n_head', 4)
        self.gate_limit = getattr(config, 'shadow_gate_limit', 0.20)
        self.loss_weight = getattr(config, 'shadow_loss_weight', 0.05)
        self.regime_weight = getattr(config, 'shadow_regime_weight', 0.02)
        self.drift_weight = getattr(config, 'shadow_drift_weight', 0.01)
        self.sparsity_weight = getattr(config, 'shadow_sparsity_weight', 0.01)
        self.provenance_weight = getattr(config, 'shadow_provenance_weight', 0.005)
        self.benchmark_regime_weight = getattr(config, 'shadow_benchmark_regime_weight', 0.50)
        self.benchmark_drift_weight = getattr(config, 'shadow_benchmark_drift_weight', 0.25)
        self.benchmark_provenance_weight = getattr(config, 'shadow_benchmark_provenance_weight', 0.25)
        self.benchmark_complexity_weight = getattr(config, 'shadow_benchmark_complexity_weight', 0.10)
        self.adapter_scale = getattr(config, 'shadow_adapter_scale', 0.12)
        self.adapter_family_scale = getattr(config, 'shadow_adapter_family_scale', 0.08)
        self.adapter_global_scale = getattr(config, 'shadow_adapter_global_scale', 0.05)
        self.adapter_promotion_rate = getattr(config, 'shadow_adapter_promotion_rate', 0.10)
        self.adapter_demotion_rate = getattr(config, 'shadow_adapter_demotion_rate', 0.08)
        self.adapter_reject_penalty = getattr(config, 'shadow_adapter_reject_penalty', 0.12)
        self.adapter_min_scale = getattr(config, 'shadow_adapter_min_scale', 0.05)
        self.adapter_branch_limit = max(1, int(getattr(config, 'shadow_adapter_branch_limit', 4)))
        self.candidate_branch_trials = max(1, int(getattr(config, 'shadow_candidate_branch_trials', 4)))
        self.candidate_branch_perturb_scale = float(getattr(config, 'shadow_candidate_branch_perturb_scale', 0.10))
        self.micro_adapter_rank = max(1, int(getattr(config, 'shadow_micro_adapter_rank', 4)))
        self.branch_learning_rate = float(getattr(config, 'shadow_branch_learning_rate', 0.18))
        self.branch_grad_decay = float(getattr(config, 'shadow_branch_grad_decay', 0.92))
        self.branch_grad_clip = float(getattr(config, 'shadow_branch_grad_clip', 0.25))
        self.branch_momentum = float(getattr(config, 'shadow_branch_momentum', 0.75))
        self.branch_optimizer_decay = float(getattr(config, 'shadow_branch_optimizer_decay', 0.98))
        self.lineage_spawn_threshold = int(getattr(config, 'shadow_lineage_spawn_threshold', 2))
        self.lineage_prune_threshold = float(getattr(config, 'shadow_lineage_prune_threshold', -0.20))
        self.lineage_max_children = int(getattr(config, 'shadow_lineage_max_children', 2))
        self.lineage_maturity_generation = max(1, int(getattr(config, 'shadow_lineage_maturity_generation', 1)))
        self.lineage_priority_boost = float(getattr(config, 'shadow_lineage_priority_boost', 0.12))
        self.lineage_search_bonus = max(0, int(getattr(config, 'shadow_lineage_search_bonus', 1)))
        self.lineage_bandwidth_bonus = max(0, int(getattr(config, 'shadow_lineage_bandwidth_bonus', 1)))
        self.lineage_mature_lr_scale = float(getattr(config, 'shadow_lineage_mature_lr_scale', 0.85))
        self.lineage_newborn_lr_scale = float(getattr(config, 'shadow_lineage_newborn_lr_scale', 1.10))
        self.lineage_mature_momentum_bonus = float(getattr(config, 'shadow_lineage_mature_momentum_bonus', 0.08))
        self.lineage_newborn_momentum_scale = float(getattr(config, 'shadow_lineage_newborn_momentum_scale', 0.92))
        self.lineage_mature_mutation_scale = float(getattr(config, 'shadow_lineage_mature_mutation_scale', 0.75))
        self.lineage_newborn_mutation_scale = float(getattr(config, 'shadow_lineage_newborn_mutation_scale', 1.35))
        self.lineage_spawn_objective_threshold = float(getattr(config, 'shadow_lineage_spawn_objective_threshold', 0.08))
        self.lineage_prune_objective_threshold = float(getattr(config, 'shadow_lineage_prune_objective_threshold', -0.05))
        self.lineage_mature_memory_horizon = max(1, int(getattr(config, 'shadow_lineage_mature_memory_horizon', 8)))
        self.lineage_newborn_memory_horizon = max(1, int(getattr(config, 'shadow_lineage_newborn_memory_horizon', 3)))
        self.lineage_mature_replay_decay = float(getattr(config, 'shadow_lineage_mature_replay_decay', 0.96))
        self.lineage_newborn_replay_decay = float(getattr(config, 'shadow_lineage_newborn_replay_decay', 0.82))
        self.lineage_fusion_similarity_threshold = float(getattr(config, 'shadow_lineage_fusion_similarity_threshold', 0.92))
        self.lineage_split_variance_threshold = float(getattr(config, 'shadow_lineage_split_variance_threshold', 0.18))
        self.lineage_fusion_priority_bonus = float(getattr(config, 'shadow_lineage_fusion_priority_bonus', 0.10))
        self.lineage_split_mutation_boost = float(getattr(config, 'shadow_lineage_split_mutation_boost', 1.25))
        self.lineage_fusion_memory_bonus = max(0, int(getattr(config, 'shadow_lineage_fusion_memory_bonus', 2)))
        self.lineage_split_memory_scale = float(getattr(config, 'shadow_lineage_split_memory_scale', 0.65))
        self.lineage_split_replay_decay_scale = float(getattr(config, 'shadow_lineage_split_replay_decay_scale', 0.88))
        self.profile_id = 'shadowhott.overlay.v0'
        self.meta = ShadowMetaTransformer(self.meta_dim, self.hidden_dim, self.depth, self.n_head, self.shadow_layers, self.gate_limit)
        self.attn_adapter_basis = nn.Parameter(torch.empty(self.micro_adapter_rank, self.shadow_layers))
        self.mlp_adapter_basis = nn.Parameter(torch.empty(self.micro_adapter_rank, self.shadow_layers))
        self.resid_adapter_basis = nn.Parameter(torch.empty(self.micro_adapter_rank, self.shadow_layers))
        self.logit_adapter_basis = nn.Parameter(torch.empty(self.micro_adapter_rank))
        nn.init.normal_(self.attn_adapter_basis, mean=0.0, std=0.05)
        nn.init.normal_(self.mlp_adapter_basis, mean=0.0, std=0.05)
        nn.init.normal_(self.resid_adapter_basis, mean=0.0, std=0.05)
        nn.init.normal_(self.logit_adapter_basis, mean=0.0, std=0.05)
        self.last_report: dict[str, Any] = {}
        self.last_certificate: dict[str, Any] = {}
        self._state = ShadowState()
        self._last_control_summary = {
            'attn_mean': 1.0,
            'mlp_mean': 1.0,
            'resid_mean': 1.0,
            'logit_scale_mean': 1.0,
            'intervention_strength': 0.0,
            'persist_score': 0.0,
        }
        self.accepted_templates: list[dict[str, Any]] = []
        self.rejected_templates: list[dict[str, Any]] = []
        self.accepted_templates_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.rejected_templates_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.acceptance_log: list[ShadowAcceptanceRecord] = []
        self._candidate_counter = 0
        self._last_candidate: dict[str, Any] | None = None
        self._persistent_prior: dict[str, torch.Tensor] = {}
        self._persistent_priors_by_signature: dict[str, dict[str, Any]] = {}
        self._persistent_adapter: dict[str, Any] = {}
        self._persistent_adapters_by_signature: dict[str, dict[str, Any]] = {}
        self._persistent_adapter_branches_global: dict[str, dict[str, Any]] = {}
        self._persistent_adapter_branches_by_signature: dict[str, dict[str, dict[str, Any]]] = {}
        self._selected_adapter_branch_global: str | None = None
        self._selected_adapter_branch_by_signature: dict[str, str] = {}
        self._adapter_stats_global: dict[str, Any] = self._new_adapter_stats()
        self._adapter_stats_by_signature: dict[str, dict[str, Any]] = {}
        self._adapter_branch_stats_global: dict[str, dict[str, Any]] = {}
        self._adapter_branch_stats_by_signature: dict[str, dict[str, dict[str, Any]]] = {}
        self._branch_learned_updates_global: dict[str, dict[str, Any]] = {}
        self._branch_learned_updates_by_signature: dict[str, dict[str, dict[str, Any]]] = {}
        self._branch_optimizer_state_global: dict[str, dict[str, Any]] = {}
        self._branch_optimizer_state_by_signature: dict[str, dict[str, dict[str, Any]]] = {}
        self._branch_lineage_global: dict[str, dict[str, Any]] = {}
        self._branch_lineage_by_signature: dict[str, dict[str, dict[str, Any]]] = {}
        self._episode_trace: list[dict[str, Any]] = []
        self._episode_active: bool = False
        self._episode_context_tag: str = 'generic'
        self._episode_counter: int = 0
        self._episode_active_signature: str | None = None
        self._episode_active_prior: dict[str, Any] | None = None
        self._episode_active_prior_source: str | None = None
        self._episode_active_prior_blend: float = 0.0

    def init_weights(self):
        self.meta.init_weights()

    def reset_state(self):
        self._state = ShadowState()
        self._last_control_summary = {
            'attn_mean': 1.0,
            'mlp_mean': 1.0,
            'resid_mean': 1.0,
            'logit_scale_mean': 1.0,
            'intervention_strength': 0.0,
            'persist_score': 0.0,
        }
        self.last_report = {}
        self.last_certificate = {}
        self.accepted_templates = []
        self.rejected_templates = []
        self.accepted_templates_by_signature = defaultdict(list)
        self.rejected_templates_by_signature = defaultdict(list)
        self.acceptance_log = []
        self._candidate_counter = 0
        self._last_candidate = None
        self._persistent_prior = {}
        self._persistent_priors_by_signature = {}
        self._persistent_adapter = {}
        self._persistent_adapters_by_signature = {}
        self._persistent_adapter_branches_global = {}
        self._persistent_adapter_branches_by_signature = {}
        self._selected_adapter_branch_global = None
        self._selected_adapter_branch_by_signature = {}
        self._adapter_stats_global = self._new_adapter_stats()
        self._adapter_stats_by_signature = {}
        self._adapter_branch_stats_global = {}
        self._adapter_branch_stats_by_signature = {}
        self._episode_trace = []
        self._episode_active = False
        self._episode_context_tag = 'generic'
        self._episode_active_signature = None
        self._episode_active_prior = None
        self._episode_active_prior_source = None
        self._episode_active_prior_blend = 0.0


    def begin_episode(self, context_tag: str | None = None) -> dict[str, Any]:
        self._episode_counter += 1
        self._episode_active = True
        self._episode_context_tag = self._normalize_context_tag(context_tag)
        self._episode_trace = []
        self._episode_active_signature = None
        self._episode_active_prior = None
        self._episode_active_prior_source = None
        self._episode_active_prior_blend = 0.0
        return {
            'episode_id': self._episode_counter,
            'context_tag': self._episode_context_tag,
        }

    def end_episode(self) -> dict[str, Any]:
        self._episode_active = False
        live_signatures = [str(step.get('live_context_signature', '')) for step in self._episode_trace]
        route_switches = 0
        for prev, cur in zip(live_signatures, live_signatures[1:]):
            if prev != cur:
                route_switches += 1
        live_reroute_activations = sum(1 for step in self._episode_trace if bool(step.get('live_reroute_applied', False)))
        gate_reroute_activations = sum(1 for step in self._episode_trace if bool(step.get('episode_gate_reroute_applied', False)))
        return {
            'episode_id': self._episode_counter,
            'context_tag': self._episode_context_tag,
            'steps': len(self._episode_trace),
            'route_switches': route_switches,
            'live_reroute_activations': live_reroute_activations,
            'gate_reroute_activations': gate_reroute_activations,
            'active_reroute_signature': self._episode_active_signature,
            'final_signature': live_signatures[-1] if live_signatures else None,
            'trace': list(self._episode_trace),
        }

    def get_episode_trace(self) -> list[dict[str, Any]]:
        return list(self._episode_trace)

    def _normalize_context_tag(self, context_tag: str | None) -> str:
        tag = (context_tag or "generic").strip().lower().replace(" ", "_")
        return tag or "generic"

    def _regime_bucket_from_state(self, state: ShadowState | None) -> str:
        if state is None:
            return "truth"
        masses = {
            "truth": float(state.truth_mass),
            "falsity": float(state.falsity_mass),
            "both": float(state.both_mass),
            "neither": float(state.neither_mass),
        }
        return max(masses, key=masses.get)

    def _provenance_bucket_from_state(self, state: ShadowState | None) -> str:
        if state is None:
            return "prov_hi"
        p = float(state.provenance_coherence)
        if p >= 0.80:
            return "prov_hi"
        if p >= 0.50:
            return "prov_mid"
        return "prov_low"

    def _contradiction_bucket_from_state(self, state: ShadowState | None) -> str:
        if state is None:
            return "contr_low"
        pressure = float(state.both_mass) + 0.5 * float(state.neither_mass)
        if pressure >= 0.55:
            return "contr_high"
        if pressure >= 0.25:
            return "contr_mid"
        return "contr_low"

    def _signature_components(self, context_tag: str | None, prev_state: ShadowState | None) -> dict[str, str]:
        task = self._normalize_context_tag(context_tag)
        regime = self._regime_bucket_from_state(prev_state)
        prov = self._provenance_bucket_from_state(prev_state)
        contr = self._contradiction_bucket_from_state(prev_state)
        return {
            'task_family': task,
            'regime_bucket': regime,
            'provenance_bucket': prov,
            'contradiction_bucket': contr,
            'family_regime_key': f"{task}|{regime}",
        }

    def _signature_components_from_observation(
        self,
        context_tag: str | None,
        state: ShadowState | None,
        four_value_histogram: dict[str, float] | None = None,
        provenance_coherence: float | None = None,
        contradiction_pressure: float | None = None,
    ) -> dict[str, str]:
        task = self._normalize_context_tag(context_tag)
        if four_value_histogram:
            masses = {
                'truth': float(four_value_histogram.get('T', 0.0)),
                'falsity': float(four_value_histogram.get('F', 0.0)),
                'both': float(four_value_histogram.get('B', 0.0)),
                'neither': float(four_value_histogram.get('N', 0.0)),
            }
            regime = max(masses, key=masses.get)
            both_mass = masses['both']
            neither_mass = masses['neither']
        else:
            regime = self._regime_bucket_from_state(state)
            both_mass = float(state.both_mass) if state is not None else 0.0
            neither_mass = float(state.neither_mass) if state is not None else 0.0
        p = float(provenance_coherence) if provenance_coherence is not None else (float(state.provenance_coherence) if state is not None else 1.0)
        if contradiction_pressure is None:
            contradiction_pressure = both_mass + 0.5 * neither_mass
        if p >= 0.80:
            prov = 'prov_hi'
        elif p >= 0.50:
            prov = 'prov_mid'
        else:
            prov = 'prov_low'
        if contradiction_pressure >= 0.55:
            contr = 'contr_high'
        elif contradiction_pressure >= 0.25:
            contr = 'contr_mid'
        else:
            contr = 'contr_low'
        return {
            'task_family': task,
            'regime_bucket': regime,
            'provenance_bucket': prov,
            'contradiction_bucket': contr,
            'family_regime_key': f"{task}|{regime}",
        }

    def current_context_signature(self, context_tag: str | None, prev_state: ShadowState | None) -> str:
        c = self._signature_components(context_tag, prev_state)
        return f"{c['task_family']}|{c['regime_bucket']}|{c['provenance_bucket']}|{c['contradiction_bucket']}"

    def context_signature_components(self, context_tag: str | None, prev_state: ShadowState | None) -> dict[str, str]:
        c = self._signature_components(context_tag, prev_state)
        c['context_signature'] = self.current_context_signature(context_tag, prev_state)
        return c

    def context_signature_components_from_report(self, context_tag: str | None, report: dict[str, Any], state: ShadowState | None = None) -> dict[str, str]:
        four = dict(report.get('four_value_histogram', {}))
        inv = dict(report.get('invariants', {}))
        contradiction_pressure = float(four.get('B', 0.0)) + 0.5 * float(four.get('N', 0.0))
        c = self._signature_components_from_observation(
            context_tag,
            state,
            four_value_histogram=four,
            provenance_coherence=float(inv.get('provenance_coherence', state.provenance_coherence if state is not None else 1.0)),
            contradiction_pressure=contradiction_pressure,
        )
        c['context_signature'] = f"{c['task_family']}|{c['regime_bucket']}|{c['provenance_bucket']}|{c['contradiction_bucket']}"
        return c

    def _clone_control_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().cpu().clone()

    def _current_template_dict(self, control: ShadowControl, context_signature: str) -> dict[str, Any]:
        parts = context_signature.split('|')
        task_family = parts[0] if len(parts) > 0 else 'generic'
        regime_bucket = parts[1] if len(parts) > 1 else 'truth'
        provenance_bucket = parts[2] if len(parts) > 2 else 'prov_hi'
        contradiction_bucket = parts[3] if len(parts) > 3 else 'contr_low'
        return {
            'candidate_id': self._candidate_counter,
            'context_signature': context_signature,
            'task_family': task_family,
            'regime_bucket': regime_bucket,
            'provenance_bucket': provenance_bucket,
            'contradiction_bucket': contradiction_bucket,
            'family_regime_key': f"{task_family}|{regime_bucket}",
            'attn_gates': self._clone_control_tensor(control.attn_gates.mean(dim=0)),
            'mlp_gates': self._clone_control_tensor(control.mlp_gates.mean(dim=0)),
            'resid_gates': self._clone_control_tensor(control.resid_gates.mean(dim=0)),
            'logit_scale': float(control.logit_scale.detach().float().mean().item()),
            'intervention_strength': float(control.intervention_strength.detach().float().mean().item()),
            'persist_score': float(control.persist_score.detach().float().mean().item()),
            'control_summary': control.summary(),
        }


    def _new_adapter_stats(self) -> dict[str, Any]:
        return {
            'quality': 0.0,
            'active_scale': 1.0,
            'accept_count': 0,
            'reject_count': 0,
            'score_sum': 0.0,
            'score_count': 0,
            'last_score_delta': 0.0,
            'suppressed': False,
        }

    def _update_adapter_stats(self, stats: dict[str, Any], score_delta: float, accepted: bool) -> dict[str, Any]:
        score = float(score_delta)
        score_mag = max(0.0, min(1.0, abs(score)))
        quality = float(stats.get('quality', 0.0))
        active_scale = float(stats.get('active_scale', 1.0))
        if accepted:
            quality = min(1.0, quality + self.adapter_promotion_rate * score_mag)
            active_scale = min(1.5, active_scale + 0.5 * self.adapter_promotion_rate * max(score_mag, self.adapter_min_scale))
            stats['accept_count'] = int(stats.get('accept_count', 0)) + 1
        else:
            penalty = self.adapter_reject_penalty if score < 0.0 else self.adapter_demotion_rate
            quality = max(0.0, quality - penalty * max(score_mag, self.adapter_min_scale))
            active_scale = max(0.0, active_scale - penalty * (0.9 + 0.6 * max(score_mag, self.adapter_min_scale)))
            stats['reject_count'] = int(stats.get('reject_count', 0)) + 1
        stats['quality'] = quality
        stats['active_scale'] = active_scale
        stats['score_sum'] = float(stats.get('score_sum', 0.0)) + score
        stats['score_count'] = int(stats.get('score_count', 0)) + 1
        stats['last_score_delta'] = score
        stats['suppressed'] = active_scale < self.adapter_min_scale
        return stats

    def _optimizer_state_template(self) -> dict[str, Any]:
        zeros = torch.zeros(self.micro_adapter_rank, dtype=torch.float32)
        return {
            'attn_velocity': zeros.clone(),
            'mlp_velocity': zeros.clone(),
            'resid_velocity': zeros.clone(),
            'logit_velocity': zeros.clone(),
            'steps': 0,
            'last_lr': 0.0,
            'last_momentum': 0.0,
            'score_sum': 0.0,
            'selection_count': 0,
        }

    def get_branch_optimizer_state(self, context_signature: str | None, branch_key: str | None) -> dict[str, Any]:
        if not branch_key:
            return self._optimizer_state_template()
        if context_signature and branch_key in self._branch_optimizer_state_by_signature.get(context_signature, {}):
            state = self._branch_optimizer_state_by_signature[context_signature][branch_key]
        else:
            state = self._branch_optimizer_state_global.get(branch_key, self._optimizer_state_template())
        out = self._optimizer_state_template()
        for k, v in state.items():
            out[k] = v.clone() if torch.is_tensor(v) else v
        return out

    def update_branch_optimizer_state(self, context_signature: str | None, branch_key: str | None, optimizer_state: dict[str, Any] | None, score_delta: float = 0.0) -> None:
        if not branch_key or optimizer_state is None:
            return
        def normalize(state: dict[str, Any]) -> dict[str, Any]:
            out = self._optimizer_state_template()
            for name in ['attn_velocity', 'mlp_velocity', 'resid_velocity', 'logit_velocity']:
                if name in state:
                    out[name] = torch.as_tensor(state[name]).detach().float().clone()
            out['steps'] = int(state.get('steps', 0))
            out['last_lr'] = float(state.get('last_lr', 0.0))
            out['last_momentum'] = float(state.get('last_momentum', self.branch_momentum))
            out['score_sum'] = float(state.get('score_sum', 0.0)) + float(score_delta)
            out['selection_count'] = int(state.get('selection_count', 0)) + 1
            return out
        norm = normalize(optimizer_state)
        self._branch_optimizer_state_global[branch_key] = norm
        if context_signature:
            sig_map = dict(self._branch_optimizer_state_by_signature.get(context_signature, {}))
            sig_map[branch_key] = norm
            self._branch_optimizer_state_by_signature[context_signature] = sig_map

    def _trajectory_bias_from_state(self, optimizer_state: dict[str, Any] | None, scale: float = 1.0) -> dict[str, torch.Tensor]:
        if not optimizer_state:
            z = torch.zeros(self.micro_adapter_rank, dtype=torch.float32)
            return {
                'attn_micro_coeffs': z.clone(),
                'mlp_micro_coeffs': z.clone(),
                'resid_micro_coeffs': z.clone(),
                'logit_micro_coeffs': z.clone(),
            }
        decay = float(self.branch_optimizer_decay)
        return {
            'attn_micro_coeffs': scale * decay * torch.as_tensor(optimizer_state.get('attn_velocity', torch.zeros(self.micro_adapter_rank))).detach().float(),
            'mlp_micro_coeffs': scale * decay * torch.as_tensor(optimizer_state.get('mlp_velocity', torch.zeros(self.micro_adapter_rank))).detach().float(),
            'resid_micro_coeffs': scale * decay * torch.as_tensor(optimizer_state.get('resid_velocity', torch.zeros(self.micro_adapter_rank))).detach().float(),
            'logit_micro_coeffs': scale * decay * torch.as_tensor(optimizer_state.get('logit_velocity', torch.zeros(self.micro_adapter_rank))).detach().float(),
        }

    def _lineage_profile(self, context_signature: str | None, branch_key: str | None, optimizer_state: dict[str, Any] | None = None) -> dict[str, Any]:
        node = self._get_lineage_node(context_signature, branch_key) if branch_key else self._lineage_node_template()
        if optimizer_state is None and branch_key is not None:
            optimizer_state = self.get_branch_optimizer_state(context_signature, branch_key)
        optimizer_state = optimizer_state or self._optimizer_state_template()
        generation = int(node.get('generation', 0))
        visits = int(node.get('visits', 0))
        selection_count = int(optimizer_state.get('selection_count', 0))
        score_count = max(1, int(node.get('score_count', 0)))
        mean_score = float(node.get('score_sum', 0.0)) / score_count
        objective_count = max(1, int(node.get('objective_count', 0)))
        objective_mean = float(node.get('objective_score_sum', 0.0)) / objective_count
        stage_count = max(1, int(node.get('stage_count', 0)))
        stage_mean = float(node.get('stage_score_sum', 0.0)) / stage_count
        mature = generation >= self.lineage_maturity_generation or selection_count > self.lineage_spawn_threshold or (visits >= self.lineage_spawn_threshold and (mean_score > 0.0 or objective_mean > 0.0))
        priority = min(self.lineage_priority_boost, 0.04 * generation + 0.02 * selection_count + 0.20 * max(0.0, mean_score))
        search_bonus = min(self.lineage_search_bonus, max(1, generation) if mature else 0)
        bandwidth_bonus = min(self.lineage_bandwidth_bonus, max(1, generation) if mature else 0)
        replay_blend_bonus = 0.10 * priority / max(self.lineage_priority_boost, 1e-6) if self.lineage_priority_boost > 0 else 0.0
        if mature:
            lr_scale = max(0.05, min(2.0, self.lineage_mature_lr_scale / (1.0 + 0.15 * generation + 0.10 * max(0.0, mean_score))))
            momentum = max(0.0, min(0.999, self.branch_momentum + self.lineage_mature_momentum_bonus + 0.02 * max(0.0, mean_score)))
            mutation_scale = max(0.10, min(2.5, self.lineage_mature_mutation_scale / (1.0 + 0.10 * generation + 0.05 * max(0.0, mean_score))))
            memory_horizon = max(1, int(self.lineage_mature_memory_horizon + min(generation, self.lineage_bandwidth_bonus + self.lineage_search_bonus)))
            replay_decay = max(0.10, min(0.999, self.lineage_mature_replay_decay - 0.01 * min(3, generation) + 0.01 * max(0.0, mean_score)))
        else:
            lr_scale = max(0.05, min(2.0, self.lineage_newborn_lr_scale))
            momentum = max(0.0, min(0.999, self.branch_momentum * self.lineage_newborn_momentum_scale))
            mutation_scale = max(0.10, min(2.5, self.lineage_newborn_mutation_scale))
            memory_horizon = max(1, int(self.lineage_newborn_memory_horizon))
            replay_decay = max(0.10, min(0.999, self.lineage_newborn_replay_decay))
        return {
            'generation': generation,
            'visits': visits,
            'mean_score': mean_score,
            'objective_mean': objective_mean,
            'stage_mean': stage_mean,
            'spawn_credit': float(node.get('spawn_credit', 0.0)),
            'prune_pressure': float(node.get('prune_pressure', 0.0)),
            'mature': bool(mature and not bool(node.get('pruned', False))),
            'priority': float(priority),
            'search_bonus': int(search_bonus),
            'bandwidth_bonus': int(bandwidth_bonus),
            'replay_blend_bonus': float(replay_blend_bonus),
            'lr_scale': float(lr_scale),
            'momentum': float(momentum),
            'mutation_scale': float(mutation_scale),
            'memory_horizon': int(memory_horizon),
            'replay_decay': float(replay_decay),
            'stage_variance': float(node.get('stage_variance', 0.0)),
            'fusion_ready': bool((mature and not bool(node.get('pruned', False))) and float(node.get('stage_variance', 0.0)) <= float(self.lineage_split_variance_threshold) and max(0, int(node.get('accepted_count', 0))) > 0),
            'split_ready': bool((not mature) and float(node.get('stage_variance', 0.0)) >= float(self.lineage_split_variance_threshold)),
            'fusion_count': int(node.get('fusion_count', 0)),
            'split_count': int(node.get('split_count', 0)),
            'pruned': bool(node.get('pruned', False)),
        }

    def _attach_lineage_profile_to_bucket(self, context_signature: str | None, branch_key: str | None, bucket: dict[str, Any]) -> dict[str, Any]:
        if not bucket:
            return bucket
        prof = self._lineage_profile(context_signature, branch_key)
        enriched = dict(bucket)
        enriched['lineage_generation'] = int(prof['generation'])
        enriched['lineage_priority'] = float(prof['priority'])
        enriched['lineage_mean_score'] = float(prof['mean_score'])
        enriched['lineage_mature'] = bool(prof['mature'])
        enriched['lineage_search_bonus'] = int(prof['search_bonus'])
        enriched['lineage_bandwidth_bonus'] = int(prof['bandwidth_bonus'])
        enriched['replay_priority'] = float(1.0 + prof['priority'])
        enriched['replay_blend_bonus'] = float(prof['replay_blend_bonus'])
        enriched['lineage_lr_scale'] = float(prof['lr_scale'])
        enriched['lineage_momentum'] = float(prof['momentum'])
        enriched['lineage_mutation_scale'] = float(prof['mutation_scale'])
        enriched['lineage_memory_horizon'] = int(prof['memory_horizon'])
        enriched['lineage_replay_decay'] = float(prof['replay_decay'])
        enriched['lineage_stage_variance'] = float(prof['stage_variance'])
        enriched['lineage_fusion_ready'] = bool(prof['fusion_ready'])
        enriched['lineage_split_ready'] = bool(prof['split_ready'])
        enriched['lineage_fusion_count'] = int(prof['fusion_count'])
        enriched['lineage_split_count'] = int(prof['split_count'])
        enriched['lineage_objective_mean'] = float(prof['objective_mean'])
        enriched['lineage_stage_mean'] = float(prof['stage_mean'])
        enriched['lineage_spawn_credit'] = float(prof['spawn_credit'])
        enriched['lineage_prune_pressure'] = float(prof['prune_pressure'])
        return enriched

    def _lineage_node_template(self, parent_key: str | None = None, generation: int = 0) -> dict[str, Any]:
        return {
            'parent_key': parent_key,
            'generation': int(generation),
            'children': [],
            'selection_count': 0,
            'score_sum': 0.0,
            'accepted_count': 0,
            'rejected_count': 0,
            'objective_score_sum': 0.0,
            'objective_count': 0,
            'stage_score_sum': 0.0,
            'stage_count': 0,
            'spawn_credit': 0.0,
            'prune_pressure': 0.0,
            'last_stage_score': 0.0,
            'stage_sq_sum': 0.0,
            'stage_variance': 0.0,
            'fusion_count': 0,
            'split_count': 0,
            'pruned': False,
        }

    def _get_lineage_node(self, context_signature: str | None, branch_key: str | None) -> dict[str, Any]:
        if not branch_key:
            return self._lineage_node_template()
        if context_signature and branch_key in self._branch_lineage_by_signature.get(context_signature, {}):
            return dict(self._branch_lineage_by_signature[context_signature][branch_key])
        if branch_key in self._branch_lineage_global:
            return dict(self._branch_lineage_global[branch_key])
        return self._lineage_node_template()

    def _set_lineage_node(self, context_signature: str | None, branch_key: str, node: dict[str, Any]) -> None:
        self._branch_lineage_global[branch_key] = dict(node)
        if context_signature:
            sig_map = dict(self._branch_lineage_by_signature.get(context_signature, {}))
            sig_map[branch_key] = dict(node)
            self._branch_lineage_by_signature[context_signature] = sig_map

    def update_branch_lineage(self, context_signature: str | None, branch_key: str | None, score_delta: float, *, accepted: bool, parent_key: str | None = None, benchmark: dict[str, Any] | None = None) -> None:
        if not branch_key:
            return
        node = self._get_lineage_node(context_signature, branch_key)
        node['selection_count'] = int(node.get('selection_count', 0)) + 1
        node['score_sum'] = float(node.get('score_sum', 0.0)) + float(score_delta)
        if accepted:
            node['accepted_count'] = int(node.get('accepted_count', 0)) + 1
        else:
            node['rejected_count'] = int(node.get('rejected_count', 0)) + 1
        gains = dict((benchmark or {}).get('selected_objective_gains', {}))
        objective_score = float((benchmark or {}).get('selected_objective_score', score_delta))
        profile_before = self._lineage_profile(context_signature, branch_key)
        optimizer_before = self.get_branch_optimizer_state(context_signature, branch_key)
        mature_stage = bool(profile_before.get('mature', False)) or bool(node.get('pruned', False)) or int(node.get('generation', 0)) >= self.lineage_maturity_generation or int(optimizer_before.get('selection_count', 0)) > self.lineage_spawn_threshold
        if mature_stage:
            stage_score = (
                float(gains.get('drift', 0.0))
                + float(gains.get('provenance', 0.0))
                + float(gains.get('consistency', 0.0))
                - 0.5 * float(gains.get('complexity_penalty', 0.0))
                + 0.2 * float(score_delta)
            )
        else:
            stage_score = (
                float(gains.get('ce', score_delta))
                + float(gains.get('regime', 0.0))
                - 0.25 * float(gains.get('complexity_penalty', 0.0))
            )
        if not accepted:
            stage_score -= 0.25 * abs(float(score_delta))
        node['objective_score_sum'] = float(node.get('objective_score_sum', 0.0)) + float(objective_score)
        node['objective_count'] = int(node.get('objective_count', 0)) + 1
        node['stage_score_sum'] = float(node.get('stage_score_sum', 0.0)) + float(stage_score)
        node['stage_count'] = int(node.get('stage_count', 0)) + 1
        node['last_stage_score'] = float(stage_score)
        node['stage_sq_sum'] = float(node.get('stage_sq_sum', 0.0)) + float(stage_score) * float(stage_score)
        node['spawn_credit'] = max(0.0, 0.85 * float(node.get('spawn_credit', 0.0)) + max(0.0, float(stage_score)))
        node['prune_pressure'] = max(0.0, 0.85 * float(node.get('prune_pressure', 0.0)) + max(0.0, -float(stage_score)))
        if parent_key is not None and node.get('parent_key') is None:
            node['parent_key'] = parent_key
            parent = self._get_lineage_node(context_signature, parent_key)
            children = list(parent.get('children', []))
            if branch_key not in children:
                children.append(branch_key)
            parent['children'] = children[: self.lineage_max_children]
            self._set_lineage_node(context_signature, parent_key, parent)
            node['generation'] = int(parent.get('generation', 0)) + 1
        mean_score = float(node['score_sum']) / max(1, int(node['selection_count']))
        objective_mean = float(node.get('objective_score_sum', 0.0)) / max(1, int(node.get('objective_count', 0)))
        stage_mean = float(node.get('stage_score_sum', 0.0)) / max(1, int(node.get('stage_count', 0)))
        stage_sq_mean = float(node.get('stage_sq_sum', 0.0)) / max(1, int(node.get('stage_count', 0)))
        node['stage_variance'] = max(0.0, stage_sq_mean - stage_mean * stage_mean)
        mature_now = mature_stage or int(node.get('generation', 0)) >= self.lineage_maturity_generation or int(optimizer_before.get('selection_count', 0)) > self.lineage_spawn_threshold
        if mature_now:
            node['pruned'] = bool(node.get('pruned', False)) or stage_mean <= float(self.lineage_prune_objective_threshold) or mean_score <= float(self.lineage_prune_threshold)
        else:
            node['pruned'] = mean_score <= float(self.lineage_prune_threshold) or (objective_mean <= float(self.lineage_prune_threshold) and int(node.get('selection_count', 0)) >= self.lineage_spawn_threshold)
        if 'fusion' in str(branch_key):
            node['fusion_count'] = int(node.get('fusion_count', 0)) + (1 if accepted else 0)
        if 'split_' in str(branch_key):
            node['split_count'] = int(node.get('split_count', 0)) + (1 if accepted else 0)
        self._set_lineage_node(context_signature, branch_key, node)

    def _trajectory_child_specs(self, template: dict[str, Any], context_signature: str, parent_branch_key: str, optimizer_state: dict[str, Any]) -> list[dict[str, Any]]:
        parent = self._get_lineage_node(context_signature, parent_branch_key)
        if int(optimizer_state.get('selection_count', 0)) < self.lineage_spawn_threshold:
            return []
        parent_prof = self._lineage_profile(context_signature, parent_branch_key, optimizer_state)
        if float(parent.get('spawn_credit', 0.0)) < float(self.lineage_spawn_objective_threshold) and not (bool(parent_prof.get('mature', False)) and float(parent_prof.get('mean_score', 0.0)) > 0.0):
            return []
        children = list(parent.get('children', []))
        prof = self._lineage_profile(context_signature, parent_branch_key, optimizer_state)
        max_children = self.lineage_max_children + int(prof.get('bandwidth_bonus', 0))
        room = max(0, max_children - len(children))
        if room <= 0:
            return []
        specs = []
        for i in range(room):
            variant = f'trajectory_child{i}'
            mode = f'candidate_branch:{variant}'
            child_tpl = dict(template)
            child_tpl['winning_mode'] = mode
            branch_key = self._adapter_branch_key(child_tpl)
            traj_scale = 0.85 + 0.10 * i
            lineage_profile = self._lineage_profile(context_signature, branch_key, optimizer_state)
            specs.append({
                'variant': variant,
                'mode': mode,
                'branch_key': branch_key,
                'prior': self._candidate_template_prior(template, variant='exact_replay'),
                'blend': float(min(0.78, 0.40 + 0.10 * i)),
                'trajectory_bias': self._trajectory_bias_from_state(optimizer_state, scale=traj_scale),
                'optimizer_state': optimizer_state,
                'trajectory_conditioned': True,
                'parent_branch_key': parent_branch_key,
                'lineage_spawned': True,
                'lineage_profile': lineage_profile,
                'replay_priority': float(1.0 + lineage_profile.get('priority', 0.0)),
                'inner_loop_steps_bonus': int(lineage_profile.get('search_bonus', 0)),
                'lineage_bandwidth_bonus': int(lineage_profile.get('bandwidth_bonus', 0)),
                'inner_loop_lr_scale': float(lineage_profile.get('lr_scale', 1.0)),
                'inner_loop_momentum': float(lineage_profile.get('momentum', self.branch_momentum)),
                'mutation_radius': float(lineage_profile.get('mutation_scale', 1.0)),
            })
        return specs

    def _make_fused_candidate_spec(self, template: dict[str, Any], context_signature: str, left_spec: dict[str, Any], right_spec: dict[str, Any]) -> dict[str, Any] | None:
        left_prof = dict(left_spec.get('lineage_profile', {}))
        right_prof = dict(right_spec.get('lineage_profile', {}))
        if not (bool(left_prof.get('fusion_ready', False)) and bool(right_prof.get('fusion_ready', False))):
            return None
        left_prior = dict(left_spec.get('prior', {}))
        right_prior = dict(right_spec.get('prior', {}))
        if not left_prior or not right_prior:
            return None
        try:
            la = torch.as_tensor(left_prior['attn_gates']).float(); ra = torch.as_tensor(right_prior['attn_gates']).float()
            lm = torch.as_tensor(left_prior['mlp_gates']).float(); rm = torch.as_tensor(right_prior['mlp_gates']).float()
            lr = torch.as_tensor(left_prior['resid_gates']).float(); rr = torch.as_tensor(right_prior['resid_gates']).float()
        except Exception:
            return None
        sim_num = torch.mean(torch.abs(la - ra)).item() + torch.mean(torch.abs(lm - rm)).item() + torch.mean(torch.abs(lr - rr)).item()
        similarity = max(0.0, 1.0 - sim_num / 3.0)
        if similarity < self.lineage_fusion_similarity_threshold:
            return None
        fused_prior = {
            'attn_gates': torch.clamp(0.5 * (la + ra), 1.0 - self.gate_limit, 1.0 + self.gate_limit),
            'mlp_gates': torch.clamp(0.5 * (lm + rm), 1.0 - self.gate_limit, 1.0 + self.gate_limit),
            'resid_gates': torch.clamp(0.5 * (lr + rr), 1.0 - self.gate_limit, 1.0 + self.gate_limit),
            'logit_scale': max(0.8, min(1.2, 0.5 * (float(left_prior.get('logit_scale', 1.0)) + float(right_prior.get('logit_scale', 1.0))))),
            'variant': 'fusion',
            'mean_intervention_strength': 0.5 * (float(left_prior.get('mean_intervention_strength', 0.0)) + float(right_prior.get('mean_intervention_strength', 0.0))),
        }
        child_template = dict(template)
        child_template['winning_mode'] = 'candidate_branch:fusion'
        child_template['regime_bucket'] = template.get('regime_bucket', 'truth')
        child_template['provenance_bucket'] = template.get('provenance_bucket', 'prov_mid')
        child_template['contradiction_bucket'] = template.get('contradiction_bucket', 'contr_low')
        branch_key = self._adapter_branch_key(child_template) + '|fusion'
        lineage_profile = self._lineage_profile(context_signature, branch_key)
        lineage_profile['priority'] = float(lineage_profile.get('priority', 0.0)) + self.lineage_fusion_priority_bonus
        return {
            'variant': 'fusion',
            'mode': 'candidate_branch:fusion',
            'branch_key': branch_key,
            'prior': fused_prior,
            'blend': max(float(left_spec.get('blend', 0.0)), float(right_spec.get('blend', 0.0))),
            'trajectory_bias': self._trajectory_bias_from_state(left_spec.get('optimizer_state', {}), scale=0.35),
            'optimizer_state': self.get_branch_optimizer_state(context_signature, branch_key),
            'trajectory_conditioned': True,
            'fusion_ready': True,
            'fused_from': [str(left_spec.get('branch_key', '')), str(right_spec.get('branch_key', ''))],
            'lineage_profile': lineage_profile,
            'replay_priority': float(1.0 + lineage_profile.get('priority', 0.0) + self.lineage_fusion_priority_bonus),
            'inner_loop_steps_bonus': int(lineage_profile.get('search_bonus', 0)),
            'lineage_bandwidth_bonus': int(lineage_profile.get('bandwidth_bonus', 0)),
            'inner_loop_lr_scale': float(lineage_profile.get('lr_scale', 1.0)),
            'inner_loop_momentum': float(lineage_profile.get('momentum', self.branch_momentum)),
            'mutation_radius': float(lineage_profile.get('mutation_scale', 1.0)),
            'memory_horizon': int(max(float(left_prof.get('memory_horizon', 1)), float(right_prof.get('memory_horizon', 1))) + self.lineage_fusion_memory_bonus),
            'replay_decay': float(max(float(left_prof.get('replay_decay', 1.0)), float(right_prof.get('replay_decay', 1.0)))),
            'memory_parent_branches': [str(left_spec.get('branch_key', '')), str(right_spec.get('branch_key', ''))],
        }

    def _make_split_candidate_specs(self, template: dict[str, Any], context_signature: str, parent_spec: dict[str, Any]) -> list[dict[str, Any]]:
        prof = dict(parent_spec.get('lineage_profile', {}))
        if not bool(prof.get('split_ready', False)):
            return []
        prior = dict(parent_spec.get('prior', {}))
        if not prior:
            return []
        mutation = max(1.0, float(parent_spec.get('mutation_radius', 1.0)) * self.lineage_split_mutation_boost)
        specs = []
        for i, variant in enumerate(['split_left', 'split_right']):
            child_template = dict(template)
            child_template['winning_mode'] = f'candidate_branch:{variant}'
            branch_key = self._adapter_branch_key(child_template)
            lineage_profile = self._lineage_profile(context_signature, branch_key)
            specs.append({
                'variant': variant,
                'mode': f'candidate_branch:{variant}',
                'branch_key': branch_key,
                'prior': self._candidate_template_prior(template, variant='sharpen' if i == 0 else 'conservative', mutation_scale=mutation),
                'blend': min(0.82, float(parent_spec.get('blend', 0.0)) + 0.05),
                'trajectory_bias': self._trajectory_bias_from_state(parent_spec.get('optimizer_state', {}), scale=(0.45 if i == 0 else -0.45)),
                'optimizer_state': self.get_branch_optimizer_state(context_signature, branch_key),
                'trajectory_conditioned': True,
                'lineage_spawned': True,
                'parent_branch_key': str(parent_spec.get('branch_key', '')),
                'split_from': str(parent_spec.get('branch_key', '')),
                'lineage_profile': lineage_profile,
                'replay_priority': float(1.0 + lineage_profile.get('priority', 0.0)),
                'inner_loop_steps_bonus': int(lineage_profile.get('search_bonus', 0)),
                'lineage_bandwidth_bonus': int(lineage_profile.get('bandwidth_bonus', 0)),
                'inner_loop_lr_scale': float(lineage_profile.get('lr_scale', 1.0)),
                'inner_loop_momentum': float(lineage_profile.get('momentum', self.branch_momentum)),
                'mutation_radius': mutation,
                'memory_horizon': max(1, int(float(prof.get('memory_horizon', 1)) * self.lineage_split_memory_scale)),
                'replay_decay': max(0.10, min(0.999, float(prof.get('replay_decay', 1.0)) * self.lineage_split_replay_decay_scale)),
                'memory_parent_branches': [str(parent_spec.get('branch_key', ''))],
            })
        return specs

    def _adapter_branch_key(self, template: dict[str, Any]) -> str:
        mode = str(template.get('winning_mode') or template.get('selected_shadow_mode') or 'base')
        regime = str(template.get('regime_bucket', 'truth'))
        prov = str(template.get('provenance_bucket', 'prov_mid'))
        contr = str(template.get('contradiction_bucket', 'contr_low'))
        return f"{mode}|{regime}|{prov}|{contr}"

    def _candidate_template_prior(self, template: dict[str, Any], variant: str = 'exact_replay', mutation_scale: float = 1.0) -> dict[str, Any]:
        attn = torch.as_tensor(template['attn_gates']).detach().float().clone()
        mlp = torch.as_tensor(template['mlp_gates']).detach().float().clone()
        resid = torch.as_tensor(template['resid_gates']).detach().float().clone()
        logit_scale = float(template.get('logit_scale', 1.0))
        strength = float(template.get('intervention_strength', 0.0))
        delta = max(0.01, float(self.candidate_branch_perturb_scale) * float(mutation_scale))
        regime = str(template.get('regime_bucket', 'truth'))
        contr = str(template.get('contradiction_bucket', 'contr_low'))
        if variant == 'exact_replay':
            pass
        elif variant == 'sharpen':
            mult = 1.0 + 0.50 * delta
            attn = 1.0 + (attn - 1.0) * mult
            mlp = 1.0 + (mlp - 1.0) * mult
            resid = 1.0 + (resid - 1.0) * mult
            logit_scale = 1.0 + (logit_scale - 1.0) * (1.0 + 0.75 * delta) + 0.10 * delta
        elif variant == 'conservative':
            mult = max(0.35, 1.0 - 0.85 * delta)
            attn = 1.0 + (attn - 1.0) * mult
            mlp = 1.0 + (mlp - 1.0) * mult
            resid = 1.0 + (resid - 1.0) * mult
            logit_scale = 1.0 + (logit_scale - 1.0) * max(0.4, 1.0 - 0.65 * delta)
        elif variant == 'contradiction_brake':
            brake = 0.55 if contr == 'contr_high' else 0.75
            attn = 1.0 + (attn - 1.0) * brake
            mlp = 1.0 + (mlp - 1.0) * brake
            resid = 1.0 + (resid - 1.0) * (0.65 if contr == 'contr_high' else 0.85)
            logit_scale = 1.0 + (logit_scale - 1.0) * 0.5
        elif variant == 'truth_push':
            push = 0.10 * delta
            if regime == 'truth':
                push *= 1.5
            elif regime == 'falsity':
                push *= 0.5
            attn = attn + push
            mlp = mlp + push
            resid = resid + 0.75 * push
            logit_scale = logit_scale + 0.12 * delta
        limit = self.gate_limit
        attn = torch.clamp(attn, 1.0 - limit, 1.0 + limit)
        mlp = torch.clamp(mlp, 1.0 - limit, 1.0 + limit)
        resid = torch.clamp(resid, 1.0 - limit, 1.0 + limit)
        logit_scale = max(0.8, min(1.2, logit_scale))
        return {
            'attn_gates': attn,
            'mlp_gates': mlp,
            'resid_gates': resid,
            'logit_scale': logit_scale,
            'variant': variant,
            'mean_intervention_strength': strength,
        }

    def candidate_branch_specs(self, template: dict[str, Any] | None, context_signature: str | None = None) -> list[dict[str, Any]]:
        if not template:
            return []
        base = ['exact_replay', 'trajectory_follow', 'trajectory_blend', 'conservative', 'sharpen', 'trajectory_counter']
        specs = []
        branch_template = dict(template)
        default_signature = context_signature or str(template.get('context_signature', 'generic|truth'))
        trajectory_parent_key = None
        trajectory_parent_state = None
        for variant in base[: max(1, self.candidate_branch_trials + 2)]:
            if variant in {'trajectory_follow', 'trajectory_blend', 'trajectory_counter'}:
                mode = f"candidate_branch:{variant}"
                branch_template['winning_mode'] = mode
                branch_key = self._adapter_branch_key(branch_template)
                lineage_node = self._get_lineage_node(default_signature, branch_key)
                if bool(lineage_node.get('pruned', False)):
                    continue
                opt_state = self.get_branch_optimizer_state(default_signature, branch_key)
                scale = 1.0 if variant == 'trajectory_follow' else (0.55 if variant == 'trajectory_blend' else -0.60)
                traj = self._trajectory_bias_from_state(opt_state, scale=scale)
                lineage_profile = self._lineage_profile(default_signature, branch_key, opt_state)
                prior = self._candidate_template_prior(template, variant='exact_replay', mutation_scale=float(lineage_profile.get('mutation_scale', 1.0)))
                if variant == 'trajectory_blend':
                    prior['variant'] = variant
                specs.append({
                    'variant': variant,
                    'mode': mode,
                    'branch_key': branch_key,
                    'prior': prior,
                    'blend': float(min(0.72, 0.34 + 0.28 * max(0.0, min(1.0, float(template.get('persist_score', 0.0)))))),
                    'trajectory_bias': traj,
                    'optimizer_state': opt_state,
                    'trajectory_conditioned': True,
                    'lineage_generation': int(lineage_node.get('generation', 0)),
                    'lineage_profile': lineage_profile,
                    'replay_priority': float(1.0 + lineage_profile.get('priority', 0.0)),
                    'inner_loop_steps_bonus': int(lineage_profile.get('search_bonus', 0)),
                    'lineage_bandwidth_bonus': int(lineage_profile.get('bandwidth_bonus', 0)),
                    'inner_loop_lr_scale': float(lineage_profile.get('lr_scale', 1.0)),
                    'inner_loop_momentum': float(lineage_profile.get('momentum', self.branch_momentum)),
                    'mutation_radius': float(lineage_profile.get('mutation_scale', 1.0)),
                })
                if variant == 'trajectory_follow':
                    trajectory_parent_key = branch_key
                    trajectory_parent_state = opt_state
                continue
            mode = f"candidate_branch:{variant}"
            branch_template = dict(template)
            branch_template['winning_mode'] = mode
            branch_key = self._adapter_branch_key(branch_template)
            lineage_node = self._get_lineage_node(default_signature, branch_key)
            if bool(lineage_node.get('pruned', False)):
                continue
            opt_state = self.get_branch_optimizer_state(default_signature, branch_key)
            lineage_profile = self._lineage_profile(default_signature, branch_key, opt_state)
            prior = self._candidate_template_prior(template, variant=variant if variant in {'exact_replay','conservative','sharpen','contradiction_brake','truth_push'} else 'exact_replay', mutation_scale=float(lineage_profile.get('mutation_scale', 1.0)))
            specs.append({
                'variant': variant,
                'mode': mode,
                'branch_key': branch_key,
                'prior': prior,
                'blend': float(min(0.65, 0.30 + 0.35 * max(0.0, min(1.0, float(template.get('persist_score', 0.0)))))),
                'trajectory_bias': self._trajectory_bias_from_state(opt_state, scale=0.15),
                'optimizer_state': opt_state,
                'trajectory_conditioned': bool(int(opt_state.get('steps', 0)) > 0),
                'lineage_generation': int(lineage_node.get('generation', 0)),
                'lineage_profile': lineage_profile,
                'replay_priority': float(1.0 + lineage_profile.get('priority', 0.0)),
                'inner_loop_steps_bonus': int(lineage_profile.get('search_bonus', 0)),
                'lineage_bandwidth_bonus': int(lineage_profile.get('bandwidth_bonus', 0)),
                'inner_loop_lr_scale': float(lineage_profile.get('lr_scale', 1.0)),
                'inner_loop_momentum': float(lineage_profile.get('momentum', self.branch_momentum)),
                'mutation_radius': float(lineage_profile.get('mutation_scale', 1.0)),
            })
        if trajectory_parent_key is not None and trajectory_parent_state is not None:
            specs.extend(self._trajectory_child_specs(template, default_signature, trajectory_parent_key, trajectory_parent_state))
        split_specs = []
        for sp in list(specs):
            split_specs.extend(self._make_split_candidate_specs(template, default_signature, sp))
        if split_specs:
            specs.extend(split_specs)
        fusion_spec = None
        fusion_candidates = [sp for sp in specs if bool(dict(sp.get('lineage_profile', {})).get('fusion_ready', False))]
        if len(fusion_candidates) >= 2:
            fusion_spec = self._make_fused_candidate_spec(template, default_signature, fusion_candidates[0], fusion_candidates[1])
        if fusion_spec is not None:
            specs.append(fusion_spec)
        specs.sort(key=lambda sp: (float(sp.get('replay_priority', 1.0)), int(sp.get('inner_loop_steps_bonus', 0))), reverse=True)
        return specs

    def _branch_selection_score(self, bucket: dict[str, Any]) -> float:
        quality = float(bucket.get('quality', 0.0))
        active_scale = float(bucket.get('active_scale', 1.0))
        mean_score = 0.0
        score_count = int(bucket.get('score_count', 0))
        if score_count > 0:
            mean_score = float(bucket.get('score_sum', 0.0)) / score_count
        template_quality = float(bucket.get('template_quality', bucket.get('quality', 0.0)))
        template_count = float(bucket.get('template_count', 0))
        novelty_bonus = min(0.10, 0.02 * template_count)
        suppression_penalty = 1.0 if bool(bucket.get('suppressed', False)) else 0.0
        lineage_bonus = float(bucket.get('lineage_priority', 0.0))
        maturity_bonus = 0.05 if bool(bucket.get('lineage_mature', False)) else 0.0
        return 0.55 * quality + 0.20 * template_quality + 0.20 * max(-1.0, min(1.0, mean_score)) + 0.10 * active_scale + novelty_bonus + lineage_bonus + maturity_bonus - suppression_penalty

    def _trim_branch_map(self, branch_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        dynamic_limit = self.adapter_branch_limit
        if branch_map:
            dynamic_limit += max(int(v.get('lineage_bandwidth_bonus', 0)) for v in branch_map.values())
        if len(branch_map) <= dynamic_limit:
            return branch_map
        items = sorted(branch_map.items(), key=lambda kv: self._branch_selection_score(kv[1]), reverse=True)
        return dict(items[:dynamic_limit])

    def _select_best_branch(self, branch_map: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
        if not branch_map:
            return None, None
        best_key = None
        best_bucket = None
        best_score = -1e9
        for key, bucket in branch_map.items():
            score = self._branch_selection_score(bucket)
            if score > best_score:
                best_score = score
                best_key = key
                best_bucket = bucket
        return best_key, best_bucket

    def _register_adapter_feedback(self, context_signature: str, score_delta: float, accepted: bool, branch_key: str | None = None) -> None:
        self._adapter_stats_global = self._update_adapter_stats(self._adapter_stats_global, score_delta, accepted)
        stats = dict(self._adapter_stats_by_signature.get(context_signature, self._new_adapter_stats()))
        self._adapter_stats_by_signature[context_signature] = self._update_adapter_stats(stats, score_delta, accepted)
        if branch_key is not None:
            gstats = dict(self._adapter_branch_stats_global.get(branch_key, self._new_adapter_stats()))
            self._adapter_branch_stats_global[branch_key] = self._update_adapter_stats(gstats, score_delta, accepted)
            sig_branch_stats = dict(self._adapter_branch_stats_by_signature.get(context_signature, {}))
            sbstats = dict(sig_branch_stats.get(branch_key, self._new_adapter_stats()))
            sig_branch_stats[branch_key] = self._update_adapter_stats(sbstats, score_delta, accepted)
            self._adapter_branch_stats_by_signature[context_signature] = sig_branch_stats

    def _attach_adapter_stats(self, bucket: dict[str, Any], stats: dict[str, Any] | None) -> dict[str, Any]:
        if not bucket:
            return bucket
        stats = stats or self._new_adapter_stats()
        enriched = dict(bucket)
        base_quality = float(enriched.get('quality', 0.0))
        stat_quality = float(stats.get('quality', 0.0))
        enriched['template_quality'] = base_quality
        enriched['quality'] = max(base_quality, stat_quality)
        enriched['active_scale'] = float(stats.get('active_scale', 1.0))
        enriched['accept_count'] = int(stats.get('accept_count', 0))
        enriched['reject_count'] = int(stats.get('reject_count', 0))
        enriched['score_sum'] = float(stats.get('score_sum', 0.0))
        enriched['score_count'] = int(stats.get('score_count', 0))
        enriched['last_score_delta'] = float(stats.get('last_score_delta', 0.0))
        enriched['suppressed'] = bool(stats.get('suppressed', False))
        return enriched

    def _make_prior_bucket(self, templates: list[dict[str, Any]], replay_decay: float = 1.0, memory_horizon: int | None = None) -> dict[str, Any]:
        if not templates:
            return {}
        replay_decay = max(0.10, min(0.999, float(replay_decay)))
        n = len(templates)
        powers = torch.arange(n - 1, -1, -1, dtype=torch.float32)
        weights = torch.pow(torch.full((n,), replay_decay, dtype=torch.float32), powers)
        weights = weights / weights.sum().clamp_min(1e-6)
        attn_stack = torch.stack([tpl['attn_gates'] for tpl in templates], dim=0).float()
        mlp_stack = torch.stack([tpl['mlp_gates'] for tpl in templates], dim=0).float()
        resid_stack = torch.stack([tpl['resid_gates'] for tpl in templates], dim=0).float()
        attn = (weights.view(n, 1) * attn_stack).sum(dim=0)
        mlp = (weights.view(n, 1) * mlp_stack).sum(dim=0)
        resid = (weights.view(n, 1) * resid_stack).sum(dim=0)
        logit_scale = float((weights * torch.tensor([float(tpl['logit_scale']) for tpl in templates], dtype=torch.float32)).sum().item())
        mode_counts: dict[str, float] = defaultdict(float)
        for w, tpl in zip(weights.tolist(), templates):
            mode = str(tpl.get('winning_mode') or tpl.get('selected_shadow_mode') or 'unknown')
            mode_counts[mode] += float(w)
        preferred_mode = max(mode_counts, key=mode_counts.get) if mode_counts else 'unknown'
        return {
            'attn_gates': attn,
            'mlp_gates': mlp,
            'resid_gates': resid,
            'logit_scale': logit_scale,
            'template_count': len(templates),
            'mode_counts': dict(mode_counts),
            'preferred_mode': preferred_mode,
            'replay_decay': float(replay_decay),
            'memory_horizon': int(memory_horizon or len(templates)),
        }


    def _project_micro_coeffs(self, delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        delta = delta.detach().float().view(-1)
        basis = basis.detach().float()
        gram = basis @ basis.transpose(0, 1)
        rhs = basis @ delta
        eye = torch.eye(gram.size(0), dtype=gram.dtype)
        coeffs = torch.linalg.solve(gram + 1e-4 * eye, rhs)
        return coeffs

    def _decode_micro_delta(self, coeffs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        coeffs = coeffs.detach().float().view(-1)
        basis = basis.detach().float()
        return coeffs @ basis

    def _bucket_micro_fields(self, attn_delta: torch.Tensor, mlp_delta: torch.Tensor, resid_delta: torch.Tensor, logit_delta: float) -> dict[str, Any]:
        attn_coeffs = self._project_micro_coeffs(attn_delta, self.attn_adapter_basis)
        mlp_coeffs = self._project_micro_coeffs(mlp_delta, self.mlp_adapter_basis)
        resid_coeffs = self._project_micro_coeffs(resid_delta, self.resid_adapter_basis)
        logit_coeffs = self._project_micro_coeffs(torch.tensor([logit_delta], dtype=torch.float32), self.logit_adapter_basis.unsqueeze(1))
        return {
            'attn_micro_coeffs': attn_coeffs,
            'mlp_micro_coeffs': mlp_coeffs,
            'resid_micro_coeffs': resid_coeffs,
            'logit_micro_coeffs': logit_coeffs,
            'micro_rank': int(self.micro_adapter_rank),
        }

    def _learned_update_template(self) -> dict[str, Any]:
        z = torch.zeros(self.micro_adapter_rank, dtype=torch.float32)
        return {
            'attn_micro_coeffs': z.clone(),
            'mlp_micro_coeffs': z.clone(),
            'resid_micro_coeffs': z.clone(),
            'logit_micro_coeffs': z.clone(),
            'learn_count': 0,
            'grad_score_sum': 0.0,
            'last_grad_scale': 0.0,
        }

    def _apply_learned_update_to_bucket(self, bucket: dict[str, Any], learned: dict[str, Any] | None) -> dict[str, Any]:
        if not bucket or not learned:
            return bucket
        out = dict(bucket)
        attn_coeffs = torch.as_tensor(out.get('attn_micro_coeffs', torch.zeros(self.micro_adapter_rank))).detach().float()
        mlp_coeffs = torch.as_tensor(out.get('mlp_micro_coeffs', torch.zeros(self.micro_adapter_rank))).detach().float()
        resid_coeffs = torch.as_tensor(out.get('resid_micro_coeffs', torch.zeros(self.micro_adapter_rank))).detach().float()
        logit_coeffs = torch.as_tensor(out.get('logit_micro_coeffs', torch.zeros(self.micro_adapter_rank))).detach().float()
        attn_coeffs = attn_coeffs + torch.as_tensor(learned.get('attn_micro_coeffs', 0.0)).detach().float()
        mlp_coeffs = mlp_coeffs + torch.as_tensor(learned.get('mlp_micro_coeffs', 0.0)).detach().float()
        resid_coeffs = resid_coeffs + torch.as_tensor(learned.get('resid_micro_coeffs', 0.0)).detach().float()
        logit_coeffs = logit_coeffs + torch.as_tensor(learned.get('logit_micro_coeffs', 0.0)).detach().float()
        out['attn_micro_coeffs'] = attn_coeffs
        out['mlp_micro_coeffs'] = mlp_coeffs
        out['resid_micro_coeffs'] = resid_coeffs
        out['logit_micro_coeffs'] = logit_coeffs
        out['attn_delta'] = out['attn_delta'] + self._decode_micro_delta(torch.as_tensor(learned.get('attn_micro_coeffs', 0.0)).float(), self.attn_adapter_basis)
        out['mlp_delta'] = out['mlp_delta'] + self._decode_micro_delta(torch.as_tensor(learned.get('mlp_micro_coeffs', 0.0)).float(), self.mlp_adapter_basis)
        out['resid_delta'] = out['resid_delta'] + self._decode_micro_delta(torch.as_tensor(learned.get('resid_micro_coeffs', 0.0)).float(), self.resid_adapter_basis)
        out['logit_delta'] = float(out.get('logit_delta', 0.0)) + float(self._decode_micro_delta(torch.as_tensor(learned.get('logit_micro_coeffs', 0.0)).float(), self.logit_adapter_basis.unsqueeze(1)).item())
        out['learn_count'] = int(learned.get('learn_count', 0))
        out['grad_score_sum'] = float(learned.get('grad_score_sum', 0.0))
        out['last_grad_scale'] = float(learned.get('last_grad_scale', 0.0))
        out['micro_rank'] = int(self.micro_adapter_rank)
        return out

    def _branch_gradient_update(self, context_signature: str, template: dict[str, Any], score_delta: float, accepted: bool, benchmark: dict[str, Any] | None = None) -> None:
        branch_key = self._adapter_branch_key(template)
        sign = 1.0 if accepted else -1.0
        magnitude = max(0.0, min(1.0, abs(float(score_delta))))
        if benchmark is not None:
            comp = float(dict(benchmark).get('composite_score_delta', score_delta))
            magnitude = max(magnitude, max(0.0, min(1.0, abs(comp))))
        grad_scale = min(self.branch_grad_clip, self.branch_learning_rate * max(magnitude, 0.05)) * sign
        attn_delta = torch.as_tensor(template['attn_gates']).detach().float() - 1.0
        mlp_delta = torch.as_tensor(template['mlp_gates']).detach().float() - 1.0
        resid_delta = torch.as_tensor(template['resid_gates']).detach().float() - 1.0
        logit_delta = float(template.get('logit_scale', 1.0)) - 1.0
        target = self._bucket_micro_fields(attn_delta, mlp_delta, resid_delta, logit_delta)

        def update_map(store: dict[str, dict[str, Any]], key: str):
            learned = dict(store.get(key, self._learned_update_template()))
            decay = float(self.branch_grad_decay)
            for name in ['attn_micro_coeffs', 'mlp_micro_coeffs', 'resid_micro_coeffs', 'logit_micro_coeffs']:
                cur = torch.as_tensor(learned.get(name, torch.zeros(self.micro_adapter_rank))).detach().float()
                tgt = torch.as_tensor(target[name]).detach().float()
                learned[name] = decay * cur + grad_scale * tgt
            learned['learn_count'] = int(learned.get('learn_count', 0)) + 1
            learned['grad_score_sum'] = float(learned.get('grad_score_sum', 0.0)) + float(score_delta)
            learned['last_grad_scale'] = float(grad_scale)
            store[key] = learned

        update_map(self._branch_learned_updates_global, branch_key)
        sig_map = dict(self._branch_learned_updates_by_signature.get(context_signature, {}))
        update_map(sig_map, branch_key)
        self._branch_learned_updates_by_signature[context_signature] = sig_map


    def _mean_score_weight(self, templates: list[dict[str, Any]]) -> float:
        if not templates:
            return 0.0
        vals = [max(0.0, float(t.get('accepted_score_delta', 0.0))) for t in templates]
        return sum(vals) / max(1, len(vals))

    def _make_adapter_bucket(self, templates: list[dict[str, Any]]) -> dict[str, Any]:
        if not templates:
            return {}
        attn = torch.stack([tpl['attn_gates'] - 1.0 for tpl in templates], dim=0).mean(dim=0)
        mlp = torch.stack([tpl['mlp_gates'] - 1.0 for tpl in templates], dim=0).mean(dim=0)
        resid = torch.stack([tpl['resid_gates'] - 1.0 for tpl in templates], dim=0).mean(dim=0)
        logit_delta = sum(float(tpl['logit_scale']) - 1.0 for tpl in templates) / len(templates)
        strength = sum(float(tpl.get('intervention_strength', 0.0)) for tpl in templates) / len(templates)
        quality = self._mean_score_weight(templates)
        mode_counts: dict[str, int] = defaultdict(int)
        for tpl in templates:
            mode = str(tpl.get('winning_mode') or tpl.get('selected_shadow_mode') or 'unknown')
            mode_counts[mode] += 1
        bucket = {
            'attn_delta': attn,
            'mlp_delta': mlp,
            'resid_delta': resid,
            'logit_delta': logit_delta,
            'template_count': len(templates),
            'mean_intervention_strength': strength,
            'quality': quality,
            'mode_counts': dict(mode_counts),
        }
        bucket.update(self._bucket_micro_fields(attn, mlp, resid, logit_delta))
        return bucket

    def _merge_adapter_buckets(self, buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
        buckets = [b for b in buckets if b]
        if not buckets:
            return None
        total_templates = sum(int(b.get('template_count', 0)) for b in buckets)
        if total_templates <= 0:
            total_templates = len(buckets)
        weights = [max(1.0, float(b.get('template_count', 1))) * max(0.25, float(b.get('active_scale', 1.0))) for b in buckets]
        denom = sum(weights)
        attn = sum(w * b['attn_delta'] for w, b in zip(weights, buckets)) / denom
        mlp = sum(w * b['mlp_delta'] for w, b in zip(weights, buckets)) / denom
        resid = sum(w * b['resid_delta'] for w, b in zip(weights, buckets)) / denom
        logit_delta = sum(w * float(b.get('logit_delta', 0.0)) for w, b in zip(weights, buckets)) / denom
        quality = sum(w * float(b.get('quality', 0.0)) for w, b in zip(weights, buckets)) / denom
        strength = sum(w * float(b.get('mean_intervention_strength', 0.0)) for w, b in zip(weights, buckets)) / denom
        active_scale = sum(w * float(b.get('active_scale', 1.0)) for w, b in zip(weights, buckets)) / denom
        accept_count = sum(int(b.get('accept_count', 0)) for b in buckets)
        reject_count = sum(int(b.get('reject_count', 0)) for b in buckets)
        score_sum = sum(float(b.get('score_sum', 0.0)) for b in buckets)
        score_count = sum(int(b.get('score_count', 0)) for b in buckets)
        suppressed = all(bool(b.get('suppressed', False)) for b in buckets)
        mode_counts: dict[str, int] = defaultdict(int)
        for b in buckets:
            for k, v in dict(b.get('mode_counts', {})).items():
                mode_counts[str(k)] += int(v)
        merged = {
            'attn_delta': attn,
            'mlp_delta': mlp,
            'resid_delta': resid,
            'logit_delta': logit_delta,
            'template_count': total_templates,
            'quality': quality,
            'mean_intervention_strength': strength,
            'active_scale': active_scale,
            'accept_count': accept_count,
            'reject_count': reject_count,
            'score_sum': score_sum,
            'score_count': score_count,
            'suppressed': suppressed,
            'mode_counts': dict(mode_counts),
        }
        merged.update(self._bucket_micro_fields(attn, mlp, resid, logit_delta))
        return merged

    def _build_branch_buckets(self, templates: list[dict[str, Any]], branch_stats_map: dict[str, dict[str, Any]] | None = None, learned_update_map: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for tpl in templates:
            grouped[self._adapter_branch_key(tpl)].append(tpl)
        branch_stats_map = branch_stats_map or {}
        learned_update_map = learned_update_map or {}
        built: dict[str, dict[str, Any]] = {}
        all_branch_keys = set(grouped.keys()) | set(learned_update_map.keys())
        for branch_key in all_branch_keys:
            lineage = self._get_lineage_node(None, branch_key)
            if bool(lineage.get('pruned', False)):
                continue
            items = grouped.get(branch_key, [])
            bucket = self._make_adapter_bucket(items) if items else {
                'attn_delta': torch.zeros(self.shadow_layers, dtype=torch.float32),
                'mlp_delta': torch.zeros(self.shadow_layers, dtype=torch.float32),
                'resid_delta': torch.zeros(self.shadow_layers, dtype=torch.float32),
                'logit_delta': 0.0,
                'template_count': 0,
                'quality': 0.0,
                'mean_intervention_strength': 0.0,
                'mode_counts': {},
            }
            bucket.update(self._bucket_micro_fields(bucket['attn_delta'], bucket['mlp_delta'], bucket['resid_delta'], float(bucket['logit_delta'])))
            bucket = self._apply_learned_update_to_bucket(bucket, learned_update_map.get(branch_key))
            if bucket:
                bucket['branch_key'] = branch_key
                if items:
                    fused_sources: list[str] = []
                    split_sources: list[str] = []
                    for tpl in items:
                        fused_sources.extend([str(x) for x in tpl.get('fused_from', []) if str(x)])
                        sf = str(tpl.get('split_from', ''))
                        if sf:
                            split_sources.append(sf)
                    if fused_sources:
                        bucket['memory_parent_branches'] = sorted(set(fused_sources))
                        bucket['memory_horizon'] = max(int(bucket.get('template_count', len(items))), max((int(t.get('memory_horizon', 1)) for t in items), default=1))
                        bucket['replay_decay'] = max(float(bucket.get('replay_decay', 1.0)), max((float(t.get('replay_decay', 1.0)) for t in items), default=1.0))
                    elif split_sources:
                        bucket['memory_parent_branches'] = [split_sources[-1]]
                        bucket['memory_horizon'] = max(1, min((int(t.get('memory_horizon', 1)) for t in items), default=1))
                        bucket['replay_decay'] = max(0.10, min(0.999, min((float(t.get('replay_decay', 1.0)) for t in items), default=1.0)))
                built[branch_key] = self._attach_adapter_stats(bucket, branch_stats_map.get(branch_key))
        return self._trim_branch_map(built)


    def _recent_templates(self, templates: list[dict[str, Any]], horizon: int | None = None) -> list[dict[str, Any]]:
        if not templates:
            return []
        if horizon is None or horizon <= 0:
            return list(templates)
        return list(templates[-int(horizon):])

    def _mean_lineage_field(self, buckets: list[dict[str, Any]], field: str, default: float) -> float:
        vals = [float(b.get(field, default)) for b in buckets if b]
        return float(sum(vals) / len(vals)) if vals else float(default)

    def _lookup_signature_prior(self, context_signature: str | None) -> tuple[dict[str, Any] | None, float, str]:
        if not context_signature:
            return None, 0.0, 'none'
        parts = context_signature.split('|')
        task_family = parts[0] if len(parts) > 0 else 'generic'
        regime_bucket = parts[1] if len(parts) > 1 else 'truth'
        provenance_bucket = parts[2] if len(parts) > 2 else 'prov_hi'

        if self._episode_active and self._episode_active_prior is not None and self._episode_active_signature is not None:
            ep_parts = self._episode_active_signature.split('|')
            if context_signature == self._episode_active_signature:
                return self._episode_active_prior, max(0.30, float(self._episode_active_prior_blend)), 'episode_exact'
            if len(ep_parts) >= 2 and len(parts) >= 2 and ep_parts[0] == task_family and ep_parts[1] == regime_bucket:
                return self._episode_active_prior, max(0.24, 0.85 * float(self._episode_active_prior_blend)), 'episode_family_regime'
            if len(ep_parts) >= 1 and ep_parts[0] == task_family:
                return self._episode_active_prior, max(0.18, 0.70 * float(self._episode_active_prior_blend)), 'episode_family'

        if context_signature in self._persistent_priors_by_signature:
            prior = self._persistent_priors_by_signature[context_signature]
            replay_bonus = float(self._persistent_adapters_by_signature.get(context_signature, {}).get('replay_blend_bonus', 0.0))
            replay_decay = float(prior.get('replay_decay', 1.0))
            return prior, min(0.55, (0.35 + replay_bonus) * replay_decay), 'exact'

        family_regime_matches = []
        family_matches = []
        for signature, bucket in self._persistent_priors_by_signature.items():
            parts2 = signature.split('|')
            if len(parts2) >= 2 and parts2[0] == task_family and parts2[1] == regime_bucket:
                family_regime_matches.append(bucket)
            if len(parts2) >= 1 and parts2[0] == task_family:
                family_matches.append(bucket)
        if family_regime_matches:
            prior = self._make_prior_bucket([
                {
                    'attn_gates': b['attn_gates'],
                    'mlp_gates': b['mlp_gates'],
                    'resid_gates': b['resid_gates'],
                    'logit_scale': b['logit_scale'],
                }
                for b in family_regime_matches
            ])
            replay_decay = self._mean_lineage_field(family_regime_matches, 'lineage_replay_decay', 1.0)
            return prior, min(0.32, 0.24 * replay_decay), 'family_regime'
        if family_matches:
            prior = self._make_prior_bucket([
                {
                    'attn_gates': b['attn_gates'],
                    'mlp_gates': b['mlp_gates'],
                    'resid_gates': b['resid_gates'],
                    'logit_scale': b['logit_scale'],
                }
                for b in family_matches
            ])
            replay_decay = self._mean_lineage_field(family_matches, 'lineage_replay_decay', 1.0)
            return prior, min(0.26, (0.18 if provenance_bucket == 'prov_hi' else 0.16) * replay_decay), 'family'
        if self._persistent_prior:
            replay_decay = float(self._persistent_prior.get('replay_decay', 1.0))
            return self._persistent_prior, min(0.18, 0.12 * replay_decay), 'global'
        return None, 0.0, 'none'

    def _lookup_signature_adapter(self, context_signature: str | None) -> tuple[dict[str, Any] | None, float, str]:
        if not context_signature:
            return None, 0.0, 'none'
        parts = context_signature.split('|')
        task_family = parts[0] if len(parts) > 0 else 'generic'
        regime_bucket = parts[1] if len(parts) > 1 else 'truth'
        if context_signature in self._persistent_adapters_by_signature:
            bucket = self._persistent_adapters_by_signature[context_signature]
            if bool(bucket.get('suppressed', False)):
                return bucket, 0.0, 'exact_suppressed'
            replay_decay = float(bucket.get('lineage_replay_decay', 1.0))
            scale = self.adapter_scale * (1.0 + 0.5 * float(bucket.get('lineage_priority', 0.0))) * replay_decay
            return bucket, scale, 'exact'
        family_regime_matches = []
        family_matches = []
        for signature, bucket in self._persistent_adapters_by_signature.items():
            parts2 = signature.split('|')
            if len(parts2) >= 2 and parts2[0] == task_family and parts2[1] == regime_bucket:
                family_regime_matches.append(bucket)
            if len(parts2) >= 1 and parts2[0] == task_family:
                family_matches.append(bucket)
        if family_regime_matches:
            merged = self._merge_adapter_buckets(family_regime_matches)
            if merged is not None:
                scale = self.adapter_family_scale * self._mean_lineage_field(family_regime_matches, 'lineage_replay_decay', 1.0)
                return merged, scale, 'family_regime'
        if family_matches:
            merged = self._merge_adapter_buckets(family_matches)
            if merged is not None:
                scale = max(0.5 * self.adapter_family_scale, 0.04) * self._mean_lineage_field(family_matches, 'lineage_replay_decay', 1.0)
                return merged, scale, 'family'
        if self._persistent_adapter:
            if bool(self._persistent_adapter.get('suppressed', False)):
                return self._persistent_adapter, 0.0, 'global_suppressed'
            scale = self.adapter_global_scale * float(self._persistent_adapter.get('lineage_replay_decay', 1.0))
            return self._persistent_adapter, scale, 'global'
        return None, 0.0, 'none'

    def _apply_adapter_bucket(self, control: ShadowControl, adapter: dict[str, Any], scale: float) -> ShadowControl:
        if not adapter or scale <= 0.0:
            return control
        device = control.attn_gates.device
        dtype = control.attn_gates.dtype
        quality = max(0.0, min(1.0, float(adapter.get('quality', 0.0))))
        strength = max(0.0, min(1.0, float(adapter.get('mean_intervention_strength', 0.0))))
        active_scale = max(0.0, min(1.5, float(adapter.get('active_scale', 1.0))))
        if bool(adapter.get('suppressed', False)) or active_scale < self.adapter_min_scale:
            return control
        eff = min(float(scale), float(scale) * active_scale * (0.5 + 0.5 * quality) * (0.75 + 0.25 * strength))
        attn_delta = adapter['attn_delta'].to(device=device, dtype=dtype).unsqueeze(0)
        mlp_delta = adapter['mlp_delta'].to(device=device, dtype=dtype).unsqueeze(0)
        resid_delta = adapter['resid_delta'].to(device=device, dtype=dtype).unsqueeze(0)
        if 'attn_micro_coeffs' in adapter:
            attn_delta = attn_delta + self._decode_micro_delta(torch.as_tensor(adapter['attn_micro_coeffs']).float(), self.attn_adapter_basis).to(device=device, dtype=dtype).unsqueeze(0)
        if 'mlp_micro_coeffs' in adapter:
            mlp_delta = mlp_delta + self._decode_micro_delta(torch.as_tensor(adapter['mlp_micro_coeffs']).float(), self.mlp_adapter_basis).to(device=device, dtype=dtype).unsqueeze(0)
        if 'resid_micro_coeffs' in adapter:
            resid_delta = resid_delta + self._decode_micro_delta(torch.as_tensor(adapter['resid_micro_coeffs']).float(), self.resid_adapter_basis).to(device=device, dtype=dtype).unsqueeze(0)
        logit_delta_val = float(adapter.get('logit_delta', 0.0))
        if 'logit_micro_coeffs' in adapter:
            logit_delta_val += float(self._decode_micro_delta(torch.as_tensor(adapter['logit_micro_coeffs']).float(), self.logit_adapter_basis.unsqueeze(1)).item())
        logit_delta = torch.tensor(logit_delta_val, device=control.logit_scale.device, dtype=control.logit_scale.dtype)
        limit = float(self.gate_limit)
        return ShadowControl(
            attn_gates=torch.clamp(control.attn_gates + eff * attn_delta, min=1.0 - limit, max=1.0 + limit),
            mlp_gates=torch.clamp(control.mlp_gates + eff * mlp_delta, min=1.0 - limit, max=1.0 + limit),
            resid_gates=torch.clamp(control.resid_gates + eff * resid_delta, min=1.0 - limit, max=1.0 + limit),
            logit_scale=torch.clamp(control.logit_scale + eff * logit_delta, min=0.8, max=1.2),
            intervention_strength=control.intervention_strength,
            persist_score=control.persist_score,
        )

    def _blend_control_with_prior(self, control: ShadowControl, prior: dict[str, Any], blend: float) -> ShadowControl:
        device = control.attn_gates.device
        dtype = control.attn_gates.dtype
        prior_attn = prior['attn_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        prior_mlp = prior['mlp_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        prior_resid = prior['resid_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        prior_logit = torch.as_tensor(prior['logit_scale'], device=control.logit_scale.device, dtype=control.logit_scale.dtype).view(() )
        return ShadowControl(
            attn_gates=(1 - blend) * control.attn_gates + blend * prior_attn,
            mlp_gates=(1 - blend) * control.mlp_gates + blend * prior_mlp,
            resid_gates=(1 - blend) * control.resid_gates + blend * prior_resid,
            logit_scale=(1 - blend) * control.logit_scale + blend * prior_logit,
            intervention_strength=control.intervention_strength,
            persist_score=control.persist_score,
        )



    def _memory_profile_from_bucket(self, bucket: dict[str, Any] | None, *, default_horizon: int, default_decay: float) -> tuple[int, float]:
        if not bucket:
            return int(default_horizon), float(default_decay)
        return max(1, int(bucket.get('memory_horizon', bucket.get('lineage_memory_horizon', default_horizon)))), max(0.10, min(0.999, float(bucket.get('replay_decay', bucket.get('lineage_replay_decay', default_decay)))))

    def _branch_templates(self, templates: list[dict[str, Any]], branch_key: str | None) -> list[dict[str, Any]]:
        if not branch_key:
            return []
        return [tpl for tpl in templates if self._adapter_branch_key(tpl) == branch_key]

    def _memory_augmented_templates(self, templates: list[dict[str, Any]], branch_key: str | None, bucket: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not templates:
            return []
        base = self._branch_templates(templates, branch_key) if branch_key else list(templates)
        base = list(base) if base else list(templates)
        if not bucket:
            return sorted(base, key=lambda t: int(t.get('candidate_id', 0)))
        parent_branches = [str(x) for x in bucket.get('memory_parent_branches', []) if str(x)]
        if parent_branches:
            seen = {int(t.get('candidate_id', -1)) for t in base}
            for parent_key in parent_branches:
                for tpl in self._branch_templates(templates, parent_key):
                    cid = int(tpl.get('candidate_id', -1))
                    if cid not in seen:
                        base.append(tpl)
                        seen.add(cid)
        return sorted(base, key=lambda t: int(t.get('candidate_id', 0)))

    def _refresh_persistent_prior(self):
        if not self.accepted_templates:
            self._persistent_prior = {}
            self._persistent_priors_by_signature = {}
            self._persistent_adapter = {}
            self._persistent_adapters_by_signature = {}
            self._persistent_adapter_branches_global = {}
            self._persistent_adapter_branches_by_signature = {}
            self._selected_adapter_branch_global = None
            self._selected_adapter_branch_by_signature = {}
            return

        self._persistent_adapter_branches_global = self._build_branch_buckets(self.accepted_templates, self._adapter_branch_stats_global, self._branch_learned_updates_global)
        self._persistent_adapter_branches_by_signature = {
            signature: self._build_branch_buckets(templates, self._adapter_branch_stats_by_signature.get(signature, {}), self._branch_learned_updates_by_signature.get(signature, {}))
            for signature, templates in self.accepted_templates_by_signature.items()
            if templates
        }
        self._selected_adapter_branch_by_signature = {}
        self._persistent_adapters_by_signature = {}
        for signature, branch_map in self._persistent_adapter_branches_by_signature.items():
            branch_key, bucket = self._select_best_branch(branch_map)
            if branch_key is not None and bucket is not None:
                bucket = self._attach_lineage_profile_to_bucket(signature, branch_key, bucket)
                self._selected_adapter_branch_by_signature[signature] = branch_key
                self._persistent_adapters_by_signature[signature] = bucket
        self._selected_adapter_branch_global, global_bucket = self._select_best_branch(self._persistent_adapter_branches_global)
        self._persistent_adapter = self._attach_lineage_profile_to_bucket(None, self._selected_adapter_branch_global, global_bucket or {}) if global_bucket is not None else {}
        global_horizon = int(self._persistent_adapter.get('memory_horizon', self._persistent_adapter.get('lineage_memory_horizon', len(self.accepted_templates) or 1))) if self._persistent_adapter else len(self.accepted_templates)
        global_decay = float(self._persistent_adapter.get('replay_decay', self._persistent_adapter.get('lineage_replay_decay', 1.0))) if self._persistent_adapter else 1.0
        global_templates = self._memory_augmented_templates(self.accepted_templates, self._selected_adapter_branch_global, self._persistent_adapter) if self._persistent_adapter else list(self.accepted_templates)
        self._persistent_prior = self._make_prior_bucket(self._recent_templates(global_templates, global_horizon), replay_decay=global_decay, memory_horizon=global_horizon)
        self._persistent_priors_by_signature = {}
        for signature, templates in self.accepted_templates_by_signature.items():
            if not templates:
                continue
            bucket = self._persistent_adapters_by_signature.get(signature, {})
            branch_key = self._selected_adapter_branch_by_signature.get(signature)
            horizon = int(bucket.get('memory_horizon', bucket.get('lineage_memory_horizon', len(templates)))) if bucket else len(templates)
            decay = float(bucket.get('replay_decay', bucket.get('lineage_replay_decay', 1.0))) if bucket else 1.0
            mem_templates = self._memory_augmented_templates(list(templates), branch_key, bucket)
            self._persistent_priors_by_signature[signature] = self._make_prior_bucket(self._recent_templates(mem_templates, horizon), replay_decay=decay, memory_horizon=horizon)

    def apply_persistent_prior(self, control: ShadowControl, context_signature: str | None = None) -> ShadowControl:
        prior, blend, _ = self._lookup_signature_prior(context_signature)
        adapted = control
        if prior is not None and blend > 0.0:
            adapted = self._blend_control_with_prior(adapted, prior, blend)
        adapter, adapter_scale, _ = self._lookup_signature_adapter(context_signature)
        if adapter is not None and adapter_scale > 0.0:
            adapted = self._apply_adapter_bucket(adapted, adapter, adapter_scale)
        return adapted

    def get_preferred_mode(self, context_signature: str | None = None) -> str:
        prior, blend, source = self._lookup_signature_prior(context_signature)
        if prior is None or blend <= 0.0:
            return 'live_only'
        mode = str(prior.get('preferred_mode') or 'live_only')
        return mode if mode != 'unknown' else 'live_only'

    def apply_preferred_mode_prior(self, control: ShadowControl, context_signature: str | None = None) -> tuple[ShadowControl, dict[str, Any]]:
        preferred_mode = self.get_preferred_mode(context_signature)
        info: dict[str, Any] = {
            'applied': False,
            'preferred_mode': preferred_mode,
            'selected_mode': 'live_only',
            'source': 'none',
            'blend': 0.0,
            'active_signature': self._episode_active_signature,
        }
        if preferred_mode == 'episode_gate' and self._episode_active:
            rerouted, gate_info = self.apply_episode_active_gate_prior(control, context_signature=context_signature)
            info.update({
                'applied': bool(gate_info.get('applied', False)),
                'selected_mode': 'episode_gate' if gate_info.get('applied', False) else 'live_only',
                'source': str(gate_info.get('source', 'none')),
                'blend': float(gate_info.get('blend', 0.0)),
                'active_signature': gate_info.get('active_signature', self._episode_active_signature),
            })
            if gate_info.get('applied', False):
                return rerouted, info
        blended = self.apply_persistent_prior(control, context_signature=context_signature)
        prior, blend, source = self._lookup_signature_prior(context_signature)
        if prior is not None and blend > 0.0:
            info.update({
                'applied': True,
                'selected_mode': 'live_only',
                'source': str(source),
                'blend': float(blend),
                'active_signature': self._episode_active_signature,
            })
        return blended, info

    def apply_episode_active_gate_prior(self, control: ShadowControl, context_signature: str | None = None) -> tuple[ShadowControl, dict[str, Any]]:
        info = {
            'applied': False,
            'source': 'none',
            'blend': 0.0,
            'active_signature': self._episode_active_signature,
        }
        if not self._episode_active:
            return control, info
        bootstrap_signature = context_signature or self._episode_active_signature
        if self._episode_active_prior is None and bootstrap_signature is not None:
            prior0, blend0, source0 = self._lookup_signature_prior(bootstrap_signature)
            if prior0 is not None and blend0 > 0.0:
                self._episode_active_signature = bootstrap_signature
                self._episode_active_prior = prior0
                self._episode_active_prior_source = f"episode_bootstrap:{source0}"
                self._episode_active_prior_blend = float(min(0.55, blend0 + 0.10))
        if self._episode_active_prior is None or self._episode_active_signature is None:
            return control, info
        prior, blend, source = self._lookup_signature_prior(context_signature)
        if prior is None or blend <= 0.0:
            return control, info
        if not str(source).startswith('episode_'):
            source = self._episode_active_prior_source or f"episode_bootstrap:{source}"
        gate_blend = min(0.55, blend + 0.10)
        device = control.attn_gates.device
        dtype = control.attn_gates.dtype
        prior_attn = prior['attn_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        prior_mlp = prior['mlp_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        prior_resid = prior['resid_gates'].to(device=device, dtype=dtype).unsqueeze(0)
        rerouted = ShadowControl(
            attn_gates=(1 - gate_blend) * control.attn_gates + gate_blend * prior_attn,
            mlp_gates=(1 - gate_blend) * control.mlp_gates + gate_blend * prior_mlp,
            resid_gates=(1 - gate_blend) * control.resid_gates + gate_blend * prior_resid,
            logit_scale=control.logit_scale,
            intervention_strength=control.intervention_strength,
            persist_score=control.persist_score,
        )
        info = {
            'applied': True,
            'source': source,
            'blend': float(gate_blend),
            'active_signature': self._episode_active_signature,
        }
        return rerouted, info

    def apply_live_episode_reroute(self, logits: torch.Tensor, targets: torch.Tensor | None, prev_state: ShadowState, control: ShadowControl, pre_context_signature: str, context_tag: str | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        state, report = self._derive_state(logits, targets, prev_state, control)
        live_components = self.context_signature_components_from_report(context_tag, report, state=state)
        live_context_signature = live_components['context_signature']
        route_info = {
            'applied': False,
            'pre_context_signature': pre_context_signature,
            'live_context_signature': live_context_signature,
            'route_source': 'none',
            'route_blend': 0.0,
            'route_logit_scale': 1.0,
        }
        if not self._episode_active:
            return logits, route_info
        if live_context_signature == pre_context_signature:
            return logits, route_info
        prior, blend, source = self._lookup_signature_prior(live_context_signature)
        if prior is None or blend <= 0.0:
            return logits, route_info
        route_blend = min(0.45, blend + 0.08)
        route_scale = 1.0 + route_blend * (float(prior['logit_scale']) - 1.0)
        rerouted_logits = logits * torch.tensor(route_scale, device=logits.device, dtype=logits.dtype)
        self._episode_active_signature = live_context_signature
        self._episode_active_prior = prior
        self._episode_active_prior_source = source
        self._episode_active_prior_blend = route_blend
        route_info = {
            'applied': True,
            'pre_context_signature': pre_context_signature,
            'live_context_signature': live_context_signature,
            'route_source': source,
            'route_blend': float(route_blend),
            'route_logit_scale': float(route_scale),
        }
        return rerouted_logits, route_info

    def accept_last_candidate(self, score_delta: float, note: str = "", winning_mode: str | None = None, benchmark: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self._last_candidate is None:
            return None
        tpl = dict(self._last_candidate)
        if winning_mode is not None:
            tpl['winning_mode'] = str(winning_mode)
        if benchmark is not None:
            tpl['benchmark'] = dict(benchmark)
        tpl['accepted_score_delta'] = float(score_delta)
        tpl['note'] = note
        self.accepted_templates.append(tpl)
        self.accepted_templates_by_signature[tpl['context_signature']].append(tpl)
        self._branch_gradient_update(str(tpl.get('context_signature', 'generic|truth')), tpl, score_delta, accepted=True, benchmark=benchmark)
        self._register_adapter_feedback(str(tpl.get('context_signature', 'generic|truth')), score_delta, accepted=True, branch_key=self._adapter_branch_key(tpl))
        self.update_branch_lineage(str(tpl.get('context_signature', 'generic|truth')), self._adapter_branch_key(tpl), score_delta, accepted=True, parent_key=tpl.get('parent_branch_key'), benchmark=benchmark)
        self._refresh_persistent_prior()
        record = ShadowAcceptanceRecord(
            candidate_id=int(tpl['candidate_id']),
            action='accept',
            score_delta=float(score_delta),
            note=note,
            context_signature=str(tpl.get('context_signature', 'generic|truth')),
            control_summary=dict(tpl['control_summary']),
            four_value_histogram=dict(self.last_report.get('four_value_histogram', {})),
        )
        self.acceptance_log.append(record)
        if self.last_certificate:
            self.last_certificate['persistence'] = {
                'status': 'accepted',
                'candidate_id': int(tpl['candidate_id']),
                'accepted_templates': len(self.accepted_templates),
                'rejected_templates': len(self.rejected_templates),
                'score_delta': float(score_delta),
                'note': note,
                'winning_mode': tpl.get('winning_mode', 'unknown'),
                'benchmark': dict(tpl.get('benchmark', {})),
                'structural_retention': {
                    'adapter_signatures': sorted(self._persistent_adapters_by_signature.keys()),
                    'adapter_count': len(self._persistent_adapters_by_signature),
                    'selected_adapter_branch_global': self._selected_adapter_branch_global,
                    'global_adapter_quality': float(self._persistent_adapter.get('quality', 0.0)) if self._persistent_adapter else 0.0,
                    'global_adapter_active_scale': float(self._persistent_adapter.get('active_scale', 0.0)) if self._persistent_adapter else 0.0,
                    'selected_adapter_branches_by_signature': dict(self._selected_adapter_branch_by_signature),
                    'learned_branch_update_signatures': sorted(self._branch_learned_updates_by_signature.keys()),
                    'global_learned_branch_updates': sorted(self._branch_learned_updates_global.keys()),
                },
            }
        self._last_candidate = None
        return tpl

    def reject_last_candidate(self, score_delta: float, note: str = "", winning_mode: str | None = None, benchmark: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self._last_candidate is None:
            return None
        tpl = dict(self._last_candidate)
        if winning_mode is not None:
            tpl['winning_mode'] = str(winning_mode)
        if benchmark is not None:
            tpl['benchmark'] = dict(benchmark)
        tpl['rejected_score_delta'] = float(score_delta)
        tpl['note'] = note
        self.rejected_templates.append(tpl)
        self.rejected_templates_by_signature[tpl['context_signature']].append(tpl)
        self._branch_gradient_update(str(tpl.get('context_signature', 'generic|truth')), tpl, score_delta, accepted=False, benchmark=benchmark)
        self._register_adapter_feedback(str(tpl.get('context_signature', 'generic|truth')), score_delta, accepted=False, branch_key=self._adapter_branch_key(tpl))
        self.update_branch_lineage(str(tpl.get('context_signature', 'generic|truth')), self._adapter_branch_key(tpl), score_delta, accepted=False, parent_key=tpl.get('parent_branch_key'), benchmark=benchmark)
        self._refresh_persistent_prior()
        record = ShadowAcceptanceRecord(
            candidate_id=int(tpl['candidate_id']),
            action='reject',
            score_delta=float(score_delta),
            note=note,
            context_signature=str(tpl.get('context_signature', 'generic|truth')),
            control_summary=dict(tpl['control_summary']),
            four_value_histogram=dict(self.last_report.get('four_value_histogram', {})),
        )
        self.acceptance_log.append(record)
        if self.last_certificate:
            self.last_certificate['persistence'] = {
                'status': 'rejected',
                'candidate_id': int(tpl['candidate_id']),
                'accepted_templates': len(self.accepted_templates),
                'rejected_templates': len(self.rejected_templates),
                'score_delta': float(score_delta),
                'note': note,
                'winning_mode': tpl.get('winning_mode', 'unknown'),
                'benchmark': dict(tpl.get('benchmark', {})),
                'structural_retention': {
                    'adapter_signatures': sorted(self._persistent_adapters_by_signature.keys()),
                    'adapter_count': len(self._persistent_adapters_by_signature),
                    'selected_adapter_branch_global': self._selected_adapter_branch_global,
                    'global_adapter_quality': float(self._persistent_adapter.get('quality', 0.0)) if self._persistent_adapter else 0.0,
                    'global_adapter_active_scale': float(self._persistent_adapter.get('active_scale', 0.0)) if self._persistent_adapter else 0.0,
                    'selected_adapter_branches_by_signature': dict(self._selected_adapter_branch_by_signature),
                    'learned_branch_update_signatures': sorted(self._branch_learned_updates_by_signature.keys()),
                    'global_learned_branch_updates': sorted(self._branch_learned_updates_global.keys()),
                },
            }
        self._last_candidate = None
        return tpl

    def _feature_vec(self, *vals: float) -> torch.Tensor:
        x = torch.tensor(vals, dtype=torch.float32)
        if x.numel() < self.meta_dim:
            x = F.pad(x, (0, self.meta_dim - x.numel()))
        return x[:self.meta_dim]

    def build_pre_tokens(self, idx: torch.Tensor, x: torch.Tensor, prev_state: ShadowState, window_left: int, in_training: bool) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        B, T, C = x.shape
        prev = prev_state.as_tensor(device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        x_det = x.detach().float()
        unique_ratio = torch.tensor([idx[b].unique().numel() / max(T, 1) for b in range(B)], device=device, dtype=torch.float32)
        input_feats = torch.stack([
            x_det.mean(dim=(1, 2)),
            x_det.std(dim=(1, 2)),
            x_det.norm(dim=-1).mean(dim=1),
            x_det.abs().mean(dim=(1, 2)),
            unique_ratio,
            torch.full((B,), T / max(T, 1), device=device, dtype=torch.float32),
            torch.full((B,), 1.0 if in_training else 0.0, device=device, dtype=torch.float32),
            torch.full((B,), prev_state.step / 1024.0, device=device, dtype=torch.float32),
        ], dim=1)
        window_feats = torch.stack([
            torch.full((B,), float(window_left), device=device, dtype=torch.float32),
            torch.full((B,), float(self.shadow_layers), device=device, dtype=torch.float32),
            torch.full((B,), float(self.gate_limit), device=device, dtype=torch.float32),
            torch.full((B,), float(C), device=device, dtype=torch.float32),
            torch.full((B,), float(T), device=device, dtype=torch.float32),
            torch.full((B,), self._last_control_summary['attn_mean'], device=device, dtype=torch.float32),
            torch.full((B,), self._last_control_summary['mlp_mean'], device=device, dtype=torch.float32),
            torch.full((B,), self._last_control_summary['logit_scale_mean'], device=device, dtype=torch.float32),
        ], dim=1)
        hist_feats = torch.stack([
            torch.full((B,), self._last_control_summary['resid_mean'], device=device, dtype=torch.float32),
            torch.full((B,), self._last_control_summary['intervention_strength'], device=device, dtype=torch.float32),
            torch.full((B,), self._last_control_summary['persist_score'], device=device, dtype=torch.float32),
            torch.full((B,), prev_state.entropy, device=device, dtype=torch.float32),
            torch.full((B,), prev_state.margin, device=device, dtype=torch.float32),
            torch.full((B,), prev_state.provenance_coherence, device=device, dtype=torch.float32),
            torch.full((B,), prev_state.mutation_risk, device=device, dtype=torch.float32),
            torch.full((B,), 1.0, device=device, dtype=torch.float32),
        ], dim=1)
        return torch.stack([prev, input_feats, window_feats, hist_feats], dim=1).to(dtype)

    def control_regularization(self, control: ShadowControl) -> torch.Tensor:
        reg = 0.0
        reg = reg + (control.attn_gates - 1.0).square().mean()
        reg = reg + (control.mlp_gates - 1.0).square().mean()
        reg = reg + (control.resid_gates - 1.0).square().mean()
        reg = reg + 0.5 * (control.logit_scale - 1.0).square().mean()
        reg = reg + 0.1 * control.intervention_strength.mean()
        return reg

    def _soft_regime_targets(self, logits: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        conf = probs.max(dim=-1).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1) / math.log(probs.size(-1))
        if targets is not None:
            mask = targets.ne(-1)
            target_probs = torch.zeros_like(conf)
            safe_targets = targets.clamp_min(0)
            gathered = probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            target_probs = torch.where(mask, gathered, target_probs)
            top1_idx = probs.argmax(dim=-1)
            correct = torch.where(mask, (top1_idx == targets).float(), torch.zeros_like(conf))
            t_score = torch.where(mask, 0.65 * target_probs + 0.35 * correct, conf)
            f_score = torch.where(mask, 0.65 * (1.0 - target_probs) + 0.35 * (1.0 - correct), entropy)
        else:
            t_score = conf
            f_score = entropy
        both_score = t_score * f_score
        neither_score = (1.0 - conf) * entropy
        scores = torch.stack([t_score, f_score, both_score, neither_score], dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return scores.mean(dim=(0, 1))

    def compute_shadow_losses(self, logits: torch.Tensor, targets: torch.Tensor | None, prev_state: ShadowState, control: ShadowControl, context_signature: str, report: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
        del context_signature
        device = logits.device
        regime_target = self._soft_regime_targets(logits, targets)
        if report is not None and 'four_value_histogram' in report:
            hist = report['four_value_histogram']
            observed = torch.tensor([
                float(hist.get('T', 0.0)),
                float(hist.get('F', 0.0)),
                float(hist.get('B', 0.0)),
                float(hist.get('N', 0.0)),
            ], device=device, dtype=torch.float32)
        else:
            observed = regime_target.detach()
        observed = observed / observed.sum().clamp_min(1e-6)
        regime = F.mse_loss(observed, regime_target)

        prev_mass = torch.tensor([
            float(prev_state.truth_mass),
            float(prev_state.falsity_mass),
            float(prev_state.both_mass),
            float(prev_state.neither_mass),
        ], device=device, dtype=torch.float32)
        prev_mass = prev_mass / prev_mass.sum().clamp_min(1e-6)
        drift_scale = 1.0 - torch.clamp((regime_target - prev_mass).abs().sum() / 2.0, 0.0, 1.0)
        drift = drift_scale * F.mse_loss(regime_target, prev_mass)

        sparsity = (
            (control.attn_gates - 1.0).abs().mean()
            + (control.mlp_gates - 1.0).abs().mean()
            + (control.resid_gates - 1.0).abs().mean()
            + 0.5 * (control.logit_scale - 1.0).abs().mean()
            + 0.25 * control.intervention_strength.mean()
        )

        prov_target = torch.clamp(1.0 - 0.5 * (regime_target[2] + regime_target[3]), 0.0, 1.0)
        truth_push = regime_target[0] + 0.5 * regime_target[2]
        provenance = (prov_target - float(prev_state.provenance_coherence)) ** 2 + 0.25 * truth_push * (1.0 - prov_target)

        consistency = F.mse_loss(torch.tensor(float(prev_state.margin), device=device), torch.clamp(regime_target[0] - regime_target[1], min=-1.0, max=1.0))
        return {
            'regime': regime,
            'drift': drift,
            'sparsity': sparsity,
            'provenance': provenance,
            'consistency': consistency,
        }

    def apply_logits(self, logits: torch.Tensor, control: ShadowControl) -> torch.Tensor:
        scale = control.logit_scale[:, None, None].to(dtype=logits.dtype)
        return logits * scale

    def _derive_state(self, logits: torch.Tensor, targets: torch.Tensor | None, prev_state: ShadowState, control: ShadowControl) -> tuple[ShadowState, dict[str, Any]]:
        with torch.no_grad():
            probs = F.softmax(logits.detach().float(), dim=-1)
            top2_vals, top2_idx = probs.topk(k=min(2, probs.size(-1)), dim=-1)
            conf = top2_vals[..., 0]
            margin = top2_vals[..., 0] - top2_vals[..., 1] if top2_vals.size(-1) > 1 else top2_vals[..., 0]
            entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1) / math.log(probs.size(-1))
            t_support = conf
            f_support = entropy
            if targets is not None:
                mask = targets.ne(-1)
                if mask.any():
                    correct = (top2_idx[..., 0] == targets) & mask
                    t_support = torch.where(mask, 0.5 * (t_support + correct.float()), t_support)
                    f_support = torch.where(mask, 0.5 * (f_support + (~correct).float()), f_support)
            t_mask = (t_support >= 0.55) & (f_support < 0.45)
            f_mask = (t_support < 0.55) & (f_support >= 0.55)
            b_mask = (t_support >= 0.55) & (f_support >= 0.55)
            n_mask = ~(t_mask | f_mask | b_mask)
            t_mass = _safe_mean(t_mask.float()).item()
            f_mass = _safe_mean(f_mask.float()).item()
            b_mass = _safe_mean(b_mask.float()).item()
            n_mass = _safe_mean(n_mask.float()).item()
            coherence_defect = b_mass + n_mass
            provenance = max(0.0, 1.0 - coherence_defect)
            mutation_risk = min(1.0, 0.5 * coherence_defect + 0.5 * abs(self._last_control_summary['attn_mean'] - 1.0) + 0.25 * float(control.persist_score.mean().item()))
            blend = 0.65 if prev_state.step > 0 else 0.0
            state = ShadowState(
                truth_mass=blend * prev_state.truth_mass + (1 - blend) * t_mass,
                falsity_mass=blend * prev_state.falsity_mass + (1 - blend) * f_mass,
                both_mass=blend * prev_state.both_mass + (1 - blend) * b_mass,
                neither_mass=blend * prev_state.neither_mass + (1 - blend) * n_mass,
                provenance_coherence=blend * prev_state.provenance_coherence + (1 - blend) * provenance,
                mutation_risk=blend * prev_state.mutation_risk + (1 - blend) * mutation_risk,
                entropy=blend * prev_state.entropy + (1 - blend) * float(entropy.mean().item()),
                margin=blend * prev_state.margin + (1 - blend) * float(margin.mean().item()),
                step=prev_state.step + 1,
            )
            determinized = 'T' if (state.truth_mass + 0.5 * state.both_mass) >= (state.falsity_mass + 0.5 * state.neither_mass) else 'F'
            report = {
                'profile_id': self.profile_id,
                'four_value_histogram': {'T': t_mass, 'F': f_mass, 'B': b_mass, 'N': n_mass},
                'bilateral_scores': {'truth': state.truth_mass, 'falsity': state.falsity_mass},
                'coarse_grain': determinized,
                'invariants': {
                    'coherence_defect': coherence_defect,
                    'provenance_coherence': state.provenance_coherence,
                    'mutation_risk': state.mutation_risk,
                    'entropy': state.entropy,
                    'margin': state.margin,
                },
                'control': control.summary(),
            }
            certificate = {
                'certificate_kind': 'shadow-certified-report',
                'profile_id': self.profile_id,
                'claims': {
                    'four_valued_runtime_overlay': True,
                    'bilateral_score_layer': True,
                    'bounded_late_layer_intervention': True,
                    'proof_carrying_rewrite_envelope': True,
                },
                'bounds': {
                    'shadow_layers': self.shadow_layers,
                    'gate_limit': self.gate_limit,
                    'logit_scale_min': 0.8,
                    'logit_scale_max': 1.2,
                },
                'determinized_readout': determinized,
            }
            report['certificate'] = certificate
            return state, report

    def update_and_report(self, logits: torch.Tensor, targets: torch.Tensor | None, prev_state: ShadowState, control: ShadowControl, record_candidate: bool = True, context_signature: str = 'generic|truth', context_tag: str | None = None, episode_step: int | None = None, sequence_len: int | None = None, live_reroute_info: dict[str, Any] | None = None) -> dict[str, Any]:
        state, report = self._derive_state(logits, targets, prev_state, control)
        pre_parts = context_signature.split('|')
        pre_components = {
            'task_family': pre_parts[0] if len(pre_parts) > 0 else 'generic',
            'regime_bucket': pre_parts[1] if len(pre_parts) > 1 else 'truth',
            'provenance_bucket': pre_parts[2] if len(pre_parts) > 2 else 'prov_hi',
            'contradiction_bucket': pre_parts[3] if len(pre_parts) > 3 else 'contr_low',
            'family_regime_key': f"{pre_parts[0] if len(pre_parts) > 0 else 'generic'}|{pre_parts[1] if len(pre_parts) > 1 else 'truth'}",
        }
        live_components = self.context_signature_components_from_report(context_tag or pre_components['task_family'], report, state=state)
        live_context_signature = live_components['context_signature']
        if record_candidate:
            self._candidate_counter += 1
            self._last_candidate = self._current_template_dict(control, context_signature=live_context_signature)
            self._last_candidate['pre_context_signature'] = context_signature
            self._last_candidate['pre_signature_components'] = dict(pre_components)
            self._last_candidate['live_context_signature'] = live_context_signature
            self._last_candidate['live_signature_components'] = dict(live_components)
        report['pre_context_signature'] = context_signature
        report['pre_context_signature_components'] = pre_components
        report['context_signature'] = live_context_signature
        report['context_signature_components'] = live_components
        report['semantic_drift'] = {
            'signature_changed': live_context_signature != context_signature,
            'from': context_signature,
            'to': live_context_signature,
        }
        live_info = dict(live_reroute_info or {'applied': False, 'route_source': 'none', 'route_blend': 0.0, 'route_logit_scale': 1.0, 'pre_context_signature': context_signature, 'live_context_signature': live_context_signature})
        gate_info = dict(live_info.pop('episode_gate_reroute', {}) or {'applied': False, 'source': 'none', 'blend': 0.0, 'active_signature': self._episode_active_signature})
        report['live_episode_reroute'] = live_info
        report['episode_gate_reroute'] = gate_info
        self._state = state
        self.last_report = report
        self.last_certificate = report['certificate']
        if record_candidate and self._last_candidate is not None:
            self.last_certificate['persistence'] = {
                'status': 'candidate',
                'candidate_id': int(self._last_candidate['candidate_id']),
                'accepted_templates': len(self.accepted_templates),
                'rejected_templates': len(self.rejected_templates),
                'persist_score': float(self._last_candidate['persist_score']),
                'context_signature': live_context_signature,
                'pre_context_signature': context_signature,
                'context_signature_components': live_components,
                'pre_context_signature_components': pre_components,
                'live_episode_reroute': report['live_episode_reroute'],
            }
        else:
            self.last_certificate['persistence'] = {
                'status': 'observation_only',
                'candidate_id': None,
                'accepted_templates': len(self.accepted_templates),
                'rejected_templates': len(self.rejected_templates),
                'context_signature': live_context_signature,
                'pre_context_signature': context_signature,
                'context_signature_components': live_components,
                'pre_context_signature_components': pre_components,
                'live_episode_reroute': report['live_episode_reroute'],
            }
        self._last_control_summary = control.summary()
        if self._episode_active:
            self._episode_trace.append({
                'episode_id': self._episode_counter,
                'episode_step': int(len(self._episode_trace) if episode_step is None else episode_step),
                'sequence_len': None if sequence_len is None else int(sequence_len),
                'pre_context_signature': context_signature,
                'live_context_signature': live_context_signature,
                'signature_changed': bool(live_context_signature != context_signature),
                'task_family': live_components['task_family'],
                'regime_bucket': live_components['regime_bucket'],
                'provenance_bucket': live_components['provenance_bucket'],
                'contradiction_bucket': live_components['contradiction_bucket'],
                'four_value_histogram': dict(report['four_value_histogram']),
                'control': dict(report['control']),
                'live_reroute_applied': bool(report['live_episode_reroute'].get('applied', False)),
                'live_reroute_source': report['live_episode_reroute'].get('route_source', 'none'),
                'live_reroute_blend': float(report['live_episode_reroute'].get('route_blend', 0.0)),
                'live_reroute_signature': report['live_episode_reroute'].get('live_context_signature', live_context_signature),
                'episode_gate_reroute_applied': bool(report['episode_gate_reroute'].get('applied', False)),
                'episode_gate_reroute_source': report['episode_gate_reroute'].get('source', 'none'),
                'episode_gate_reroute_blend': float(report['episode_gate_reroute'].get('blend', 0.0)),
                'episode_gate_reroute_signature': report['episode_gate_reroute'].get('active_signature', live_context_signature),
            })
        return report

    def get_acceptance_summary(self) -> dict[str, Any]:
        return {
            'accepted_templates': len(self.accepted_templates),
            'rejected_templates': len(self.rejected_templates),
            'pending_candidate': None if self._last_candidate is None else int(self._last_candidate['candidate_id']),
            'pending_context_signature': None if self._last_candidate is None else str(self._last_candidate.get('context_signature', 'generic|truth|prov_hi|contr_low')),
            'pending_pre_context_signature': None if self._last_candidate is None else str(self._last_candidate.get('pre_context_signature', 'generic|truth|prov_hi|contr_low')),
            'pending_signature_components': None if self._last_candidate is None else {
                'task_family': self._last_candidate.get('task_family', 'generic'),
                'regime_bucket': self._last_candidate.get('regime_bucket', 'truth'),
                'provenance_bucket': self._last_candidate.get('provenance_bucket', 'prov_hi'),
                'contradiction_bucket': self._last_candidate.get('contradiction_bucket', 'contr_low'),
                'family_regime_key': self._last_candidate.get('family_regime_key', 'generic|truth'),
            },
            'accepted_candidate_ids': [int(t['candidate_id']) for t in self.accepted_templates],
            'rejected_candidate_ids': [int(t['candidate_id']) for t in self.rejected_templates],
            'accepted_by_signature': {k: len(v) for k, v in self.accepted_templates_by_signature.items()},
            'rejected_by_signature': {k: len(v) for k, v in self.rejected_templates_by_signature.items()},
            'accepted_mode_counts': dict(self._persistent_prior.get('mode_counts', {})) if self._persistent_prior else {},
            'preferred_winning_mode': self._persistent_prior.get('preferred_mode', 'unknown') if self._persistent_prior else 'unknown',
            'persistent_prior_signatures': sorted(self._persistent_priors_by_signature.keys()),
            'preferred_mode_by_signature': {k: v.get('preferred_mode', 'unknown') for k, v in self._persistent_priors_by_signature.items()},
            'persistent_adapter_signatures': sorted(self._persistent_adapters_by_signature.keys()),
            'selected_adapter_branch_global': self._selected_adapter_branch_global,
            'selected_adapter_branch_by_signature': dict(self._selected_adapter_branch_by_signature),
            'adapter_branch_candidates_global': sorted(self._persistent_adapter_branches_global.keys()),
            'adapter_branch_candidates_by_signature': {k: sorted(v.keys()) for k, v in self._persistent_adapter_branches_by_signature.items()},
            'adapter_template_counts_by_signature': {k: int(v.get('template_count', 0)) for k, v in self._persistent_adapters_by_signature.items()},
            'adapter_quality_by_signature': {k: float(v.get('quality', 0.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'adapter_active_scale_by_signature': {k: float(v.get('active_scale', 0.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'suppressed_adapter_signatures': sorted([k for k, v in self._persistent_adapters_by_signature.items() if bool(v.get('suppressed', False))]),
            'global_adapter_template_count': int(self._persistent_adapter.get('template_count', 0)) if self._persistent_adapter else 0,
            'global_adapter_quality': float(self._persistent_adapter.get('quality', 0.0)) if self._persistent_adapter else 0.0,
            'global_adapter_active_scale': float(self._persistent_adapter.get('active_scale', 0.0)) if self._persistent_adapter else 0.0,
            'learned_branch_update_signatures': sorted(self._branch_learned_updates_by_signature.keys()),
            'global_learned_branch_updates': sorted(self._branch_learned_updates_global.keys()),
            'branch_last_grad_scale_by_signature': {k: float(v.get(self._selected_adapter_branch_by_signature.get(k, ''), {}).get('last_grad_scale', 0.0)) for k, v in self._branch_learned_updates_by_signature.items()},
            'branch_optimizer_state_signatures': sorted(self._branch_optimizer_state_by_signature.keys()),
            'global_optimizer_state_branches': sorted(self._branch_optimizer_state_global.keys()),
            'branch_lineage_signatures': sorted(self._branch_lineage_by_signature.keys()),
            'global_lineage_branches': sorted(self._branch_lineage_global.keys()),
            'pruned_lineage_branches': sorted([k for k, v in self._branch_lineage_global.items() if bool(v.get('pruned', False))]),
            'lineage_replay_priority_by_signature': {k: float(v.get('replay_priority', 1.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_bandwidth_bonus_by_signature': {k: int(v.get('lineage_bandwidth_bonus', 0)) for k, v in self._persistent_adapters_by_signature.items()},
            'global_lineage_replay_priority': float(self._persistent_adapter.get('replay_priority', 1.0)) if self._persistent_adapter else 1.0,
            'global_lineage_bandwidth_bonus': int(self._persistent_adapter.get('lineage_bandwidth_bonus', 0)) if self._persistent_adapter else 0,
            'lineage_lr_scale_by_signature': {k: float(v.get('lineage_lr_scale', 1.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_momentum_by_signature': {k: float(v.get('lineage_momentum', self.branch_momentum)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_mutation_scale_by_signature': {k: float(v.get('lineage_mutation_scale', 1.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_memory_horizon_by_signature': {k: int(v.get('lineage_memory_horizon', 0)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_replay_decay_by_signature': {k: float(v.get('lineage_replay_decay', 1.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'global_lineage_lr_scale': float(self._persistent_adapter.get('lineage_lr_scale', 1.0)) if self._persistent_adapter else 1.0,
            'global_lineage_momentum': float(self._persistent_adapter.get('lineage_momentum', self.branch_momentum)) if self._persistent_adapter else self.branch_momentum,
            'global_lineage_mutation_scale': float(self._persistent_adapter.get('lineage_mutation_scale', 1.0)) if self._persistent_adapter else 1.0,
            'global_lineage_memory_horizon': int(self._persistent_adapter.get('lineage_memory_horizon', 0)) if self._persistent_adapter else 0,
            'global_lineage_replay_decay': float(self._persistent_adapter.get('lineage_replay_decay', 1.0)) if self._persistent_adapter else 1.0,
            'lineage_stage_variance_by_signature': {k: float(v.get('lineage_stage_variance', 0.0)) for k, v in self._persistent_adapters_by_signature.items()},
            'lineage_split_readiness_by_signature': {k: bool(v.get('lineage_split_ready', False)) for k, v in self._persistent_adapters_by_signature.items()},
            'fused_adapter_branches_by_signature': {k: str(v.get('branch_key', '')) for k, v in self._persistent_adapters_by_signature.items() if 'fusion' in str(v.get('branch_key', ''))},
            'global_fused_adapter_branch': str(self._selected_adapter_branch_global) if self._selected_adapter_branch_global and 'fusion' in str(self._selected_adapter_branch_global) else '',
            'memory_parent_branches_by_signature': {k: list(v.get('memory_parent_branches', [])) for k, v in self._persistent_adapters_by_signature.items() if v.get('memory_parent_branches')},
            'global_memory_parent_branches': list(self._persistent_adapter.get('memory_parent_branches', [])) if self._persistent_adapter else [],
            'episode_active_signature': self._episode_active_signature,
            'episode_active_prior_source': self._episode_active_prior_source,
            'episode_active_prior_blend': float(self._episode_active_prior_blend),
            'log_entries': [asdict(r) for r in self.acceptance_log[-16:]],
        }

    def get_state(self) -> ShadowState:
        return self._state
