import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.gpt import GPT, GPTConfig


def test_shadowhott_overlay_forward_and_report():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    logits = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)
    report = model.get_shadow_report()
    assert report["profile_id"] == "shadowhott.overlay.v0"
    assert set(report["four_value_histogram"].keys()) == {"T", "F", "B", "N"}
    assert "bilateral_scores" in report
    assert report["certificate"]["claims"]["bounded_late_layer_intervention"] is True


def test_shadowhott_accept_reject_persistence_cycle():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    _ = model(idx)
    summary = model.get_shadow_acceptance_summary()
    assert summary["pending_candidate"] is not None
    accepted = model.shadow_accept_last(0.25, note="improved stability")
    assert accepted is not None
    summary = model.get_shadow_acceptance_summary()
    assert summary["accepted_templates"] == 1
    assert summary["pending_candidate"] is None
    _ = model(idx)
    summary = model.get_shadow_acceptance_summary()
    assert summary["pending_candidate"] is not None
    rejected = model.shadow_reject_last(-0.10, note="regressed")
    assert rejected is not None
    summary = model.get_shadow_acceptance_summary()
    assert summary["rejected_templates"] == 1


def test_shadowhott_auto_benchmark_accept_reject():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    accept_result = model.shadow_benchmark_batch(idx, targets, accept_threshold=-1.0, reject_threshold=999.0, note_prefix='forced_accept')
    assert accept_result['decision'] == 'accept'
    assert accept_result['acceptance_summary']['accepted_templates'] == 1

    reject_result = model.shadow_benchmark_batch(idx, targets, accept_threshold=999.0, reject_threshold=-1.0, note_prefix='forced_reject')
    assert reject_result['decision'] == 'reject'
    assert reject_result['acceptance_summary']['rejected_templates'] == 1


def test_shadowhott_context_sensitive_persistence_routing():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)

    # Create and accept a coding regime template.
    _ = model(idx, shadow_context_tag='coding')
    accepted = model.shadow_accept_last(0.5, note='coding win')
    assert accepted is not None
    summary = model.get_shadow_acceptance_summary()
    coding_signature = accepted['context_signature']
    assert summary['accepted_by_signature'][coding_signature] == 1

    # Matching signature should get the stronger signature-specific prior blend.
    control = model.shadowhott.meta(model.shadowhott.build_pre_tokens(idx, model.transformer.wte(idx).to(next(model.parameters()).dtype), model.shadowhott.get_state(), model.window_sizes[-1][0], in_training=False))
    matched = model.shadowhott.apply_persistent_prior(control, context_signature=coding_signature)
    parts = coding_signature.split('|')
    family_regime = '|'.join(parts[:2])
    family_regime_fallback = model.shadowhott.apply_persistent_prior(control, context_signature=f'{family_regime}|prov_low|contr_high')
    family_only_fallback = model.shadowhott.apply_persistent_prior(control, context_signature='coding|falsity|prov_low|contr_high')

    matched_gap = (matched.attn_gates - control.attn_gates).abs().mean().item()
    family_regime_gap = (family_regime_fallback.attn_gates - control.attn_gates).abs().mean().item()
    family_only_gap = (family_only_fallback.attn_gates - control.attn_gates).abs().mean().item()
    assert matched_gap > family_regime_gap > family_only_gap

    # New candidate should carry its own richer context signature.
    _ = model(idx, shadow_context_tag='math')
    summary = model.get_shadow_acceptance_summary()
    assert summary['pending_context_signature'].startswith('math|')
    assert summary['pending_signature_components']['task_family'] == 'math'
    assert summary['pending_signature_components']['provenance_bucket'].startswith('prov_')
    assert summary['pending_signature_components']['contradiction_bucket'].startswith('contr_')


def test_shadowhott_report_includes_signature_components():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    _ = model(idx, shadow_context_tag='proof')
    report = model.get_shadow_report()
    comps = report['context_signature_components']
    assert comps['task_family'] == 'proof'
    assert comps['regime_bucket'] in {'truth', 'falsity', 'both', 'neither'}
    assert comps['provenance_bucket'].startswith('prov_')
    assert comps['contradiction_bucket'].startswith('contr_')


def test_shadowhott_live_regime_signature_overrides_prepass_bucket():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
    )
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    shadow._state = shadow._state.__class__(
        truth_mass=0.05,
        falsity_mass=0.90,
        both_mass=0.03,
        neither_mass=0.02,
        provenance_coherence=0.30,
        mutation_risk=0.10,
        entropy=0.70,
        margin=0.05,
        step=3,
    )
    prev = shadow.get_state()
    pre_sig = shadow.current_context_signature('coding', prev)
    B, T, V = 1, 4, cfg.vocab_size
    logits = torch.full((B, T, V), -8.0)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    for t in range(T):
        logits[0, t, int(targets[0, t])] = 8.0
    control = shadow.meta(torch.zeros((1, 4, shadow.meta_dim), dtype=torch.float32))
    report = shadow.update_and_report(logits, targets, prev, control, record_candidate=True, context_signature=pre_sig, context_tag='coding')
    summary = shadow.get_acceptance_summary()
    assert summary['pending_pre_context_signature'] == pre_sig
    assert summary['pending_context_signature'].startswith('coding|truth|')
    assert summary['pending_context_signature'] != pre_sig
    assert report['semantic_drift']['signature_changed'] is True


def test_shadowhott_generate_episode_records_within_episode_rerouting_trace():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
    )
    model = GPT(cfg)
    model.init_weights()
    result = model.generate_episode([1, 2, 3], max_tokens=3, temperature=0.0, shadow_context_tag='coding')
    assert len(result['generated_tokens']) == 3
    episode = result['shadow_episode']
    assert episode['steps'] >= 4
    assert len(episode['trace']) == episode['steps']
    assert episode['trace'][0]['episode_step'] == 0
    assert episode['trace'][-1]['episode_step'] == episode['steps'] - 1
    assert all(step['task_family'] == 'coding' for step in episode['trace'])
    assert episode['final_signature'] == episode['trace'][-1]['live_context_signature']


def test_shadowhott_live_episode_reroute_updates_active_signature_and_trace():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
    )
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    # Seed an accepted template in a truth regime for coding.
    shadow.accepted_templates = [{
        'candidate_id': 1,
        'context_signature': 'coding|truth|prov_mid|contr_low',
        'task_family': 'coding',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'family_regime_key': 'coding|truth',
        'attn_gates': torch.ones(1),
        'mlp_gates': torch.ones(1),
        'resid_gates': torch.ones(1),
        'logit_scale': 1.2,
        'intervention_strength': 0.5,
        'persist_score': 0.5,
        'control_summary': {},
    }]
    shadow.accepted_templates_by_signature['coding|truth|prov_mid|contr_low'].append(shadow.accepted_templates[0])
    shadow._refresh_persistent_prior()
    shadow.begin_episode('coding')
    prev = shadow._state.__class__(
        truth_mass=0.05,
        falsity_mass=0.90,
        both_mass=0.03,
        neither_mass=0.02,
        provenance_coherence=0.30,
        mutation_risk=0.10,
        entropy=0.70,
        margin=0.05,
        step=3,
    )
    shadow._state = prev
    pre_sig = shadow.current_context_signature('coding', prev)
    B, T, V = 1, 4, cfg.vocab_size
    logits = torch.full((B, T, V), -8.0)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    for t in range(T):
        logits[0, t, int(targets[0, t])] = 8.0
    control = shadow.meta(torch.zeros((1, 4, shadow.meta_dim), dtype=torch.float32))
    rerouted_logits, route_info = shadow.apply_live_episode_reroute(logits, targets, prev, control, pre_context_signature=pre_sig, context_tag='coding')
    assert route_info['applied'] is True
    assert shadow._episode_active_signature == 'coding|truth|prov_mid|contr_low'
    assert route_info['route_source'] in {'exact', 'episode_exact'}
    assert not torch.equal(rerouted_logits, logits)
    report = shadow.update_and_report(rerouted_logits, targets, prev, control, record_candidate=False, context_signature=pre_sig, context_tag='coding', episode_step=0, sequence_len=T, live_reroute_info=route_info)
    episode = shadow.end_episode()
    assert episode['live_reroute_activations'] >= 1
    assert episode['active_reroute_signature'] == 'coding|truth|prov_mid|contr_low'
    assert episode['trace'][0]['live_reroute_applied'] is True
    assert report['live_episode_reroute']['applied'] is True


def test_shadowhott_episode_active_gate_prior_modulates_subsequent_steps():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
    )
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    tpl = {
        'candidate_id': 1,
        'context_signature': 'coding|truth|prov_mid|contr_low',
        'task_family': 'coding',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'family_regime_key': 'coding|truth',
        'attn_gates': torch.tensor([1.35]),
        'mlp_gates': torch.tensor([0.85]),
        'resid_gates': torch.tensor([1.25]),
        'logit_scale': 1.15,
        'intervention_strength': 0.5,
        'persist_score': 0.5,
        'control_summary': {},
    }
    shadow.accepted_templates = [tpl]
    shadow.accepted_templates_by_signature[tpl['context_signature']].append(tpl)
    shadow._refresh_persistent_prior()
    shadow.begin_episode('coding')
    shadow._episode_active_signature = tpl['context_signature']
    shadow._episode_active_prior = shadow._persistent_priors_by_signature[tpl['context_signature']]
    shadow._episode_active_prior_source = 'episode_exact'
    shadow._episode_active_prior_blend = 0.28
    control = shadow.meta(torch.zeros((1, 4, shadow.meta_dim), dtype=torch.float32))
    before_attn = control.attn_gates.clone()
    before_mlp = control.mlp_gates.clone()
    rerouted, info = shadow.apply_episode_active_gate_prior(control, context_signature=tpl['context_signature'])
    assert info['applied'] is True
    assert info['source'].startswith('episode_')
    assert (rerouted.attn_gates - before_attn).abs().mean().item() > 0.0
    assert (rerouted.mlp_gates - before_mlp).abs().mean().item() > 0.0

    idx = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    _ = model(idx, shadow_context_tag='coding', shadow_record_candidate=False, shadow_apply_persistent_prior=True, shadow_episode_step=1)
    report = model.get_shadow_report()
    assert report['episode_gate_reroute']['applied'] is True
    assert report['episode_gate_reroute']['source'].startswith('episode_')
    episode = shadow.end_episode()
    assert episode['gate_reroute_activations'] >= 1
    assert episode['trace'][-1]['episode_gate_reroute_applied'] is True


def test_shadowhott_benchmark_selects_episode_gate_when_prior_helps():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
    )
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    tpl = {
        'candidate_id': 1,
        'context_signature': 'coding|truth|prov_mid|contr_low',
        'task_family': 'coding',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'family_regime_key': 'coding|truth',
        'attn_gates': torch.tensor([1.35]),
        'mlp_gates': torch.tensor([0.85]),
        'resid_gates': torch.tensor([1.25]),
        'logit_scale': 1.10,
        'intervention_strength': 0.5,
        'persist_score': 0.5,
        'control_summary': {},
    }
    shadow.accepted_templates = [tpl]
    shadow.accepted_templates_by_signature[tpl['context_signature']].append(tpl)
    shadow._refresh_persistent_prior()

    orig_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if kwargs.get('targets') is not None or (len(args) > 1 and args[1] is not None):
            if kwargs.get('shadow_enabled_override') is False:
                return out + out.new_tensor(0.30)
            if kwargs.get('shadow_apply_episode_gate_prior', True):
                return out - out.new_tensor(0.20)
            return out + out.new_tensor(0.10)
        return out
    model.forward = wrapped_forward

    idx = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    result = model.shadow_benchmark_batch(idx, targets, accept_threshold=-1.0, reject_threshold=999.0, shadow_context_tag='coding', note_prefix='mode_select')
    assert result['selected_shadow_mode'] == 'episode_gate'
    assert result['shadow_gate_loss'] < result['shadow_live_loss']
    assert result['acceptance_summary']['accepted_templates'] >= 1


def test_shadowhott_benchmark_persists_winning_mode_and_summary():
    cfg = GPTConfig(sequence_len=8, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, shadow_enabled=True, shadow_layers=1)
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    tpl = {
        'candidate_id': 1,
        'context_signature': 'coding|truth|prov_mid|contr_low',
        'task_family': 'coding',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'family_regime_key': 'coding|truth',
        'attn_gates': torch.tensor([1.30]),
        'mlp_gates': torch.tensor([0.90]),
        'resid_gates': torch.tensor([1.20]),
        'logit_scale': 1.08,
        'intervention_strength': 0.4,
        'persist_score': 0.6,
        'control_summary': {},
        'winning_mode': 'episode_gate',
    }
    shadow.accepted_templates = [tpl]
    shadow.accepted_templates_by_signature[tpl['context_signature']].append(tpl)
    shadow._refresh_persistent_prior()

    orig_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if kwargs.get('targets') is not None or (len(args) > 1 and args[1] is not None):
            if kwargs.get('shadow_enabled_override') is False:
                return out + out.new_tensor(0.30)
            if kwargs.get('shadow_apply_episode_gate_prior', True):
                return out - out.new_tensor(0.22)
            return out + out.new_tensor(0.08)
        return out
    model.forward = wrapped_forward

    idx = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    result = model.shadow_benchmark_batch(idx, targets, accept_threshold=-1.0, reject_threshold=999.0, shadow_context_tag='coding', note_prefix='mode_persist')
    record = result['record']
    assert result['selected_shadow_mode'] == 'episode_gate'
    assert record['winning_mode'] == 'episode_gate'
    assert record['benchmark']['selected_shadow_mode'] == 'episode_gate'
    summary = result['acceptance_summary']
    assert summary['preferred_winning_mode'] == 'episode_gate'
    assert summary['accepted_mode_counts']['episode_gate'] >= 1
    assert summary['preferred_mode_by_signature'][record['context_signature']] == 'episode_gate'


def test_shadowhott_preferred_mode_replay_bootstraps_episode_gate():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
    )
    model = GPT(cfg)
    model.init_weights()
    shadow = model.shadowhott
    assert shadow is not None

    signature = 'coding|truth|prov_mid|contr_low'
    tpl = {
        'candidate_id': 1,
        'context_signature': signature,
        'task_family': 'coding',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'family_regime_key': 'coding|truth',
        'attn_gates': torch.tensor([1.30, 1.10]),
        'mlp_gates': torch.tensor([0.90, 1.15]),
        'resid_gates': torch.tensor([1.20, 1.05]),
        'logit_scale': 1.08,
        'intervention_strength': 0.4,
        'persist_score': 0.6,
        'control_summary': {},
        'winning_mode': 'episode_gate',
    }
    shadow.accepted_templates = [tpl]
    shadow.accepted_templates_by_signature[signature].append(tpl)
    shadow._refresh_persistent_prior()

    shadow.begin_episode('coding')
    control = shadow.meta(torch.zeros(1, 4, shadow.meta_dim))
    rerouted, info = shadow.apply_preferred_mode_prior(control, context_signature=signature)
    assert info['preferred_mode'] == 'episode_gate'
    assert info['selected_mode'] == 'episode_gate'
    assert info['applied'] is True
    assert str(info['source']).startswith('episode_')
    assert (rerouted.attn_gates - control.attn_gates).abs().mean().item() > 0.0



def test_shadowhott_adapter_promotion_and_rejection_suppression():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
        shadow_adapter_promotion_rate=0.4,
        shadow_adapter_reject_penalty=0.8,
        shadow_adapter_min_scale=0.2,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)

    _ = model(idx, shadow_context_tag='coding')
    accepted = model.shadow_accept_last(0.75, note='strong coding win')
    assert accepted is not None
    sig = accepted['context_signature']
    summary = model.get_shadow_acceptance_summary()
    assert summary['adapter_quality_by_signature'][sig] > 0.0
    assert summary['adapter_active_scale_by_signature'][sig] > 1.0

    _ = model(idx, shadow_context_tag='coding')
    rejected = model.shadow_reject_last(-1.0, note='strong regression')
    assert rejected is not None
    summary = model.get_shadow_acceptance_summary()
    assert sig in summary['suppressed_adapter_signatures']
    assert summary['adapter_active_scale_by_signature'][sig] < cfg.shadow_adapter_min_scale


def test_shadowhott_selects_best_adapter_branch_within_signature():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
        shadow_adapter_branch_limit=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)

    _ = model(idx, shadow_context_tag='coding')
    first = model.shadow_accept_last(0.10, note='weak branch', winning_mode='persistent_prior')
    assert first is not None
    sig = first['context_signature']

    _ = model(idx, shadow_context_tag='coding')
    second = model.shadow_accept_last(0.65, note='strong branch', winning_mode='episode_gate')
    assert second is not None
    assert second['context_signature'] == sig

    summary = model.get_shadow_acceptance_summary()
    branches = summary['adapter_branch_candidates_by_signature'][sig]
    assert len(branches) >= 2
    selected = summary['selected_adapter_branch_by_signature'][sig]
    assert selected.startswith('episode_gate|')


def test_shadowhott_branch_limit_keeps_top_branches():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
        shadow_adapter_branch_limit=2,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    modes = [
        ('persistent_prior', 0.10),
        ('episode_gate', 0.55),
        ('live_only', 0.35),
    ]
    for mode, score in modes:
        _ = model(idx, shadow_context_tag='coding')
        accepted = model.shadow_accept_last(score, note=mode, winning_mode=mode)
        assert accepted is not None
    summary = model.get_shadow_acceptance_summary()
    sig = next(iter(summary['selected_adapter_branch_by_signature'].keys()))
    branches = summary['adapter_branch_candidates_by_signature'][sig]
    assert len(branches) == 2
    assert any(branch.startswith('episode_gate|') for branch in branches)


def test_shadowhott_benchmark_candidate_branch_search_records_trials_and_selects_branch():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
        shadow_candidate_branch_trials=4,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    result = model.shadow_benchmark_batch(idx, targets, accept_threshold=-1.0, reject_threshold=999.0, note_prefix='candidate_search')
    bench = result['benchmark']
    assert len(bench['candidate_branch_trials']) >= 1
    if bench['candidate_branch_selected'] is not None:
        assert bench['candidate_branch_selected']['mode'].startswith('candidate_branch:')
    rec = result['record']
    assert rec is not None
    accepted = model.get_shadow_acceptance_summary()
    assert accepted['accepted_templates'] >= 1


def test_shadowhott_branch_local_learning_updates_micro_coeffs():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        shadow_enabled=True,
        shadow_layers=2,
        shadow_hidden_dim=64,
        shadow_n_head=4,
        shadow_micro_adapter_rank=3,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    _ = model(idx, shadow_context_tag='coding')
    accepted = model.shadow_accept_last(0.4, note='branch learn win', winning_mode='candidate_branch:sharpen', benchmark={'composite_score_delta': 0.35})
    assert accepted is not None
    summary = model.get_shadow_acceptance_summary()
    sig = accepted['context_signature']
    assert sig in summary['learned_branch_update_signatures']
    bucket = model.shadowhott._persistent_adapters_by_signature[sig]
    assert 'attn_micro_coeffs' in bucket
    assert torch.as_tensor(bucket['attn_micro_coeffs']).abs().sum().item() > 0.0
    assert summary['branch_last_grad_scale_by_signature'][sig] > 0.0


def test_shadow_benchmark_inner_loop_candidate_metadata():
    import torch
    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=16, shadowhott_enabled=True, shadow_layers=1, shadow_candidate_branch_trials=3, shadow_inner_loop_steps=1, shadow_inner_loop_lr=0.05)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    result = model.shadow_benchmark_batch(idx, targets, accept_threshold=1e9, reject_threshold=1e9, shadow_context_tag='test')
    assert result['enabled'] is True
    assert 'candidate_branch_results' in result
    assert result['candidate_branch_trials'] >= 0
    if result['candidate_branch_results']:
        first = result['candidate_branch_results'][0]
        assert 'steps_run' in first
        assert 'mode' in first



def test_shadowhott_trajectory_conditioned_candidate_specs_use_optimizer_state():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
        shadow_candidate_branch_trials=2,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    _ = model(idx, shadow_context_tag='coding')
    tpl = model.shadowhott._last_candidate
    assert tpl is not None
    sig = str(tpl.get('context_signature', 'coding|truth'))
    mode = 'candidate_branch:trajectory_follow'
    branch_tpl = dict(tpl)
    branch_tpl['winning_mode'] = mode
    branch_key = model.shadowhott._adapter_branch_key(branch_tpl)
    state = model.shadowhott.get_branch_optimizer_state(sig, branch_key)
    state['attn_velocity'] = torch.ones_like(state['attn_velocity']) * 0.2
    state['steps'] = 3
    model.shadowhott.update_branch_optimizer_state(sig, branch_key, state, score_delta=0.1)
    specs = model.shadowhott.candidate_branch_specs(tpl, context_signature=sig)
    traj_specs = [sp for sp in specs if sp.get('trajectory_conditioned')]
    assert traj_specs
    follow = [sp for sp in specs if sp['variant'] == 'trajectory_follow'][0]
    assert int(follow['optimizer_state']['steps']) >= 3
    assert torch.norm(torch.as_tensor(follow['trajectory_bias']['attn_micro_coeffs']).float()).item() > 0.0


def test_shadowhott_benchmark_persists_selected_branch_optimizer_state():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
        shadow_candidate_branch_trials=2,
        shadow_inner_loop_steps=1,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    result = model.shadow_benchmark_batch(idx, targets, accept_threshold=10.0, reject_threshold=10.0, shadow_context_tag='coding')
    summary = result['acceptance_summary']
    assert 'global_optimizer_state_branches' in summary
    if result.get('selected_candidate_type') == 'inner_loop_branch':
        assert int(result.get('selected_optimizer_steps', 0)) >= 1
        assert summary['global_optimizer_state_branches']


def test_shadowhott_lineage_children_spawn_from_optimizer_history():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
        shadow_candidate_branch_trials=2,
        shadow_lineage_spawn_threshold=1,
        shadow_lineage_max_children=2,
    )
    model = GPT(cfg)
    model.init_weights()
    idx = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    _ = model(idx, shadow_context_tag='coding')
    tpl = model.shadowhott._last_candidate
    sig = str(tpl.get('context_signature', 'coding|truth'))
    parent_mode = 'candidate_branch:trajectory_follow'
    parent_tpl = dict(tpl)
    parent_tpl['winning_mode'] = parent_mode
    parent_key = model.shadowhott._adapter_branch_key(parent_tpl)
    state = model.shadowhott.get_branch_optimizer_state(sig, parent_key)
    state['attn_velocity'] = torch.ones_like(state['attn_velocity']) * 0.1
    state['selection_count'] = 1
    model.shadowhott.update_branch_optimizer_state(sig, parent_key, state, score_delta=0.2)
    model.shadowhott.update_branch_lineage(sig, parent_key, 0.2, accepted=True)
    specs = model.shadowhott.candidate_branch_specs(tpl, context_signature=sig)
    assert any(sp.get('lineage_spawned', False) for sp in specs)


def test_shadowhott_lineage_prune_marks_bad_branch():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        shadow_enabled=True,
        shadow_layers=1,
        shadow_hidden_dim=32,
        shadow_n_head=2,
        shadow_lineage_prune_threshold=-0.05,
    )
    model = GPT(cfg)
    model.init_weights()
    sig = 'coding|truth|prov_hi|contr_clean'
    model.shadowhott.update_branch_lineage(sig, 'candidate_branch:trajectory_follow|truth|prov_hi|contr_clean', -0.2, accepted=False)
    summary = model.get_shadow_acceptance_summary()
    assert summary['pruned_lineage_branches']


def test_shadowhott_mature_lineage_gets_replay_priority_and_search_bonus():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_candidate_branch_trials=2,
        shadow_lineage_spawn_threshold=1,
        shadow_lineage_maturity_generation=1,
        shadow_lineage_priority_boost=0.2,
        shadow_lineage_search_bonus=2,
    )
    model = GPT(cfg)
    sig = 'task|truth|prov_hi|contr_clean'
    tpl = {
        'candidate_id': 1, 'context_signature': sig, 'winning_mode': 'candidate_branch:trajectory_follow',
        'persist_score': 0.8, 'task_family': 'task', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
        'attn_gates': torch.ones(model.shadowhott.shadow_layers), 'mlp_gates': torch.ones(model.shadowhott.shadow_layers),
        'resid_gates': torch.ones(model.shadowhott.shadow_layers), 'logit_scale': 1.0,
    }
    key = model.shadowhott._adapter_branch_key(tpl)
    model.shadowhott.update_branch_optimizer_state(sig, key, {'steps': 3, 'selection_count': 3}, score_delta=0.3)
    model.shadowhott.update_branch_lineage(sig, key, 0.3, accepted=True)
    specs = model.shadowhott.candidate_branch_specs(tpl, context_signature=sig)
    follow = [sp for sp in specs if sp['branch_key'] == key][0]
    assert follow['replay_priority'] > 1.0
    assert follow['inner_loop_steps_bonus'] >= 1


def test_shadowhott_mature_lineage_expands_retention_bandwidth():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_adapter_branch_limit=1,
        shadow_lineage_bandwidth_bonus=1,
        shadow_lineage_maturity_generation=1,
    )
    model = GPT(cfg)
    mature = {'quality': 0.4, 'lineage_bandwidth_bonus': 1, 'lineage_mature': True}
    newborn = {'quality': 0.3, 'lineage_bandwidth_bonus': 0, 'lineage_mature': False}
    trimmed = model.shadowhott._trim_branch_map({'a': mature, 'b': newborn})
    assert len(trimmed) == 2


def test_shadowhott_lineage_hyperparameters_shape_candidate_optimizer_policy():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_candidate_branch_trials=2,
        shadow_lineage_spawn_threshold=1,
        shadow_lineage_maturity_generation=1,
        shadow_lineage_search_bonus=1,
        shadow_lineage_mature_lr_scale=0.8,
        shadow_lineage_newborn_lr_scale=1.2,
        shadow_lineage_mature_momentum_bonus=0.1,
        shadow_lineage_newborn_momentum_scale=0.9,
        shadow_lineage_mature_mutation_scale=0.7,
        shadow_lineage_newborn_mutation_scale=1.4,
        shadow_branch_momentum=0.7,
    )
    model = GPT(cfg)
    sig = 'hyper|truth|prov_hi|contr_clean'
    tpl = {
        'candidate_id': 1, 'context_signature': sig, 'winning_mode': 'candidate_branch:trajectory_follow',
        'persist_score': 0.8, 'task_family': 'hyper', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
        'attn_gates': torch.ones(model.shadowhott.shadow_layers), 'mlp_gates': torch.ones(model.shadowhott.shadow_layers),
        'resid_gates': torch.ones(model.shadowhott.shadow_layers), 'logit_scale': 1.0,
    }
    key = model.shadowhott._adapter_branch_key(tpl)
    newborn = [sp for sp in model.shadowhott.candidate_branch_specs(tpl, context_signature=sig) if sp['branch_key'] == key][0]
    assert newborn['inner_loop_lr_scale'] > 1.0
    assert newborn['inner_loop_momentum'] < cfg.shadow_branch_momentum
    assert newborn['mutation_radius'] > 1.0
    model.shadowhott.update_branch_optimizer_state(sig, key, {'steps': 3, 'selection_count': 3}, score_delta=0.3)
    model.shadowhott.update_branch_lineage(sig, key, 0.3, accepted=True)
    mature = [sp for sp in model.shadowhott.candidate_branch_specs(tpl, context_signature=sig) if sp['branch_key'] == key][0]
    assert mature['inner_loop_lr_scale'] < 1.0
    assert mature['inner_loop_momentum'] > cfg.shadow_branch_momentum
    assert mature['mutation_radius'] < 1.0


def test_shadowhott_lineage_mutation_radius_scales_candidate_perturbation():
    cfg = GPTConfig(vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
        shadow_candidate_branch_perturb_scale=0.2,
        shadow_lineage_mature_mutation_scale=0.6,
        shadow_lineage_newborn_mutation_scale=1.5,
    )
    overlay = GPT(cfg).shadowhott
    template = {
        'attn_gates': torch.tensor([1.1, 0.9], dtype=torch.float32),
        'mlp_gates': torch.tensor([1.1, 0.9], dtype=torch.float32),
        'resid_gates': torch.tensor([1.05, 0.95], dtype=torch.float32),
        'logit_scale': 1.05,
        'intervention_strength': 0.4,
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_mid',
        'contradiction_bucket': 'contr_low',
        'persist_score': 0.5,
        'context_signature': 'sig|truth',
    }
    newborn = overlay._candidate_template_prior(template, variant='sharpen', mutation_scale=cfg.shadow_lineage_newborn_mutation_scale)
    mature = overlay._candidate_template_prior(template, variant='sharpen', mutation_scale=cfg.shadow_lineage_mature_mutation_scale)
    newborn_delta = float(torch.mean(torch.abs(newborn['attn_gates'] - 1.0)))
    mature_delta = float(torch.mean(torch.abs(mature['attn_gates'] - 1.0)))
    assert newborn_delta > mature_delta


def test_shadowhott_lineage_objective_weighting_differs_for_mature_and_newborn():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_mature_ce_weight=0.8,
        shadow_lineage_newborn_ce_weight=1.2,
        shadow_lineage_mature_stability_weight=1.3,
        shadow_lineage_newborn_stability_weight=0.7,
        shadow_lineage_mature_coherence_weight=1.25,
        shadow_lineage_newborn_coherence_weight=0.8,
    )
    model = GPT(cfg)
    baseline = {
        'ce_loss': 2.0,
        'regime_loss': 0.30,
        'drift_loss': 0.20,
        'provenance_loss': 0.25,
        'consistency_loss': 0.20,
        'sparsity_loss': 0.10,
        'control_regularization': 0.01,
    }
    candidate = {
        'ce_loss': 1.95,
        'regime_loss': 0.27,
        'drift_loss': 0.08,
        'provenance_loss': 0.10,
        'consistency_loss': 0.08,
        'sparsity_loss': 0.07,
        'control_regularization': 0.03,
    }
    newborn_score, newborn_meta = model._shadow_objective_score(baseline, candidate, {'mature': False})
    mature_score, mature_meta = model._shadow_objective_score(baseline, candidate, {'mature': True})
    assert newborn_meta['weights']['ce'] > mature_meta['weights']['ce']
    assert newborn_meta['weights']['drift'] < mature_meta['weights']['drift']
    assert newborn_meta['weights']['provenance'] < mature_meta['weights']['provenance']
    assert mature_score > newborn_score


def test_shadowhott_lineage_spawn_and_prune_use_stage_objective_criteria():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_spawn_threshold=2,
        shadow_lineage_spawn_objective_threshold=0.05,
        shadow_lineage_prune_objective_threshold=-0.02,
    )
    model = GPT(cfg)
    overlay = model.shadowhott
    sig = 'sig|truth'
    newborn_key = 'candidate_branch:truth_push|truth|prov_hi|contr_clean'
    mature_key = 'candidate_branch:trajectory_follow|truth|prov_hi|contr_clean'

    overlay.update_branch_optimizer_state(sig, newborn_key, {'selection_count': 0}, score_delta=0.1)
    overlay.update_branch_lineage(sig, newborn_key, 0.06, accepted=True, benchmark={
        'selected_objective_score': 0.10,
        'selected_objective_gains': {'ce': 0.08, 'regime': 0.03, 'complexity_penalty': 0.0},
    })
    newborn_node = overlay._get_lineage_node(sig, newborn_key)
    assert float(newborn_node.get('spawn_credit', 0.0)) >= cfg.shadow_lineage_spawn_objective_threshold
    template = {
        'attn_gates': torch.ones(cfg.shadow_layers),
        'mlp_gates': torch.ones(cfg.shadow_layers),
        'resid_gates': torch.ones(cfg.shadow_layers),
        'logit_scale': 1.0,
        'intervention_strength': 0.2,
        'winning_mode': 'candidate_branch:truth_push',
        'regime_bucket': 'truth',
        'provenance_bucket': 'prov_hi',
        'contradiction_bucket': 'contr_clean',
        'context_signature': sig,
    }
    specs = overlay._trajectory_child_specs(template, sig, newborn_key, {'selection_count': 2})
    assert specs, 'positive newborn stage-objective should unlock lineage children'

    overlay.update_branch_optimizer_state(sig, mature_key, {'selection_count': 3}, score_delta=0.1)
    overlay.update_branch_lineage(sig, mature_key, 0.02, accepted=True, benchmark={
        'selected_objective_score': -0.08,
        'selected_objective_gains': {'drift': -0.03, 'provenance': -0.03, 'consistency': -0.03, 'complexity_penalty': 0.0},
    })
    overlay.update_branch_lineage(sig, mature_key, 0.01, accepted=False, benchmark={
        'selected_objective_score': -0.10,
        'selected_objective_gains': {'drift': -0.03, 'provenance': -0.04, 'consistency': -0.03, 'complexity_penalty': 0.0},
    })
    mature_node = overlay._get_lineage_node(sig, mature_key)
    assert bool(mature_node.get('pruned', False))


def test_shadowhott_lineage_memory_horizon_and_replay_decay_differ_by_stage():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_spawn_threshold=1,
        shadow_lineage_maturity_generation=1,
        shadow_lineage_mature_memory_horizon=7,
        shadow_lineage_newborn_memory_horizon=2,
        shadow_lineage_mature_replay_decay=0.97,
        shadow_lineage_newborn_replay_decay=0.75,
    )
    model = GPT(cfg)
    sig = 'mem|truth|prov_hi|contr_clean'
    tpl = {
        'candidate_id': 1, 'context_signature': sig, 'winning_mode': 'candidate_branch:trajectory_follow',
        'persist_score': 0.8, 'task_family': 'mem', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
        'attn_gates': torch.ones(model.shadowhott.shadow_layers), 'mlp_gates': torch.ones(model.shadowhott.shadow_layers),
        'resid_gates': torch.ones(model.shadowhott.shadow_layers), 'logit_scale': 1.0,
    }
    key = model.shadowhott._adapter_branch_key(tpl)
    newborn_prof = model.shadowhott._lineage_profile(sig, key, {'selection_count': 0})
    assert newborn_prof['memory_horizon'] == cfg.shadow_lineage_newborn_memory_horizon
    assert abs(newborn_prof['replay_decay'] - cfg.shadow_lineage_newborn_replay_decay) < 1e-6
    model.shadowhott.update_branch_optimizer_state(sig, key, {'selection_count': 3}, score_delta=0.3)
    model.shadowhott.update_branch_lineage(sig, key, 0.3, accepted=True)
    mature_prof = model.shadowhott._lineage_profile(sig, key)
    assert mature_prof['memory_horizon'] > newborn_prof['memory_horizon']
    assert mature_prof['replay_decay'] > newborn_prof['replay_decay']


def test_shadowhott_refresh_persistent_prior_uses_lineage_memory_horizon_and_decay():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_mature_memory_horizon=4,
        shadow_lineage_newborn_memory_horizon=2,
        shadow_lineage_mature_replay_decay=0.98,
        shadow_lineage_newborn_replay_decay=0.70,
    )
    overlay = GPT(cfg).shadowhott
    sig = 'sig|truth|prov_hi|contr_clean'
    templates = []
    for i, val in enumerate([1.00, 1.10, 1.20, 1.30, 1.40], start=1):
        templates.append({
            'candidate_id': i,
            'context_signature': sig,
            'winning_mode': 'candidate_branch:trajectory_follow',
            'attn_gates': torch.full((cfg.shadow_layers,), val),
            'mlp_gates': torch.full((cfg.shadow_layers,), val),
            'resid_gates': torch.full((cfg.shadow_layers,), val),
            'logit_scale': val,
            'accepted_score_delta': 0.05,
            'intervention_strength': 0.2,
            'persist_score': 0.5,
            'task_family': 'sig', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
        })
    overlay.accepted_templates = list(templates)
    overlay.accepted_templates_by_signature[sig] = list(templates)
    branch_key = overlay._adapter_branch_key(templates[-1])
    bucket = {
        'quality': 0.5,
        'active_scale': 1.0,
        'mean_intervention_strength': 0.2,
        'lineage_priority': 0.0,
        'lineage_mature': False,
        'lineage_memory_horizon': 2,
        'lineage_replay_decay': 0.70,
    }
    overlay._persistent_adapter_branches_by_signature = {sig: {branch_key: bucket}}
    overlay._persistent_adapters_by_signature = {sig: bucket}
    overlay._selected_adapter_branch_by_signature = {sig: branch_key}
    overlay._persistent_adapter_branches_global = {branch_key: dict(bucket, lineage_memory_horizon=4, lineage_replay_decay=0.98)}
    overlay._selected_adapter_branch_global = branch_key
    overlay._persistent_adapter = dict(bucket, lineage_memory_horizon=4, lineage_replay_decay=0.98)
    overlay._branch_lineage_by_signature[sig] = {branch_key: overlay._lineage_node_template(generation=0)}
    overlay._branch_lineage_global[branch_key] = overlay._lineage_node_template(generation=2)
    overlay._refresh_persistent_prior()
    prior_sig = overlay._persistent_priors_by_signature[sig]
    assert prior_sig['template_count'] == 2
    assert prior_sig['memory_horizon'] == 2
    assert abs(prior_sig['replay_decay'] - 0.70) < 1e-6
    prior_global = overlay._persistent_prior
    assert prior_global['template_count'] == 5
    assert prior_global['memory_horizon'] == 6
    assert abs(prior_global['replay_decay'] - 0.96) < 1e-6


def test_shadowhott_fused_branch_consolidates_parent_memory_templates():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_fusion_memory_bonus=2,
    )
    overlay = GPT(cfg).shadowhott
    sig = 'fus|truth|prov_hi|contr_clean'
    def mk(i, mode, val, **extra):
        return {
            'candidate_id': i,
            'context_signature': sig,
            'winning_mode': mode,
            'attn_gates': torch.full((cfg.shadow_layers,), val),
            'mlp_gates': torch.full((cfg.shadow_layers,), val),
            'resid_gates': torch.full((cfg.shadow_layers,), val),
            'logit_scale': val,
            'accepted_score_delta': 0.2,
            'intervention_strength': 0.2,
            'persist_score': 0.5,
            'task_family': 'fus', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
            **extra,
        }
    left = mk(1, 'candidate_branch:left', 1.05)
    right = mk(2, 'candidate_branch:right', 1.15)
    fused = mk(3, 'candidate_branch:fusion', 1.10, fused_from=[overlay._adapter_branch_key(left), overlay._adapter_branch_key(right)], memory_horizon=4, replay_decay=0.95, accepted_score_delta=0.45)
    overlay.accepted_templates = [left, right, fused]
    overlay.accepted_templates_by_signature[sig] = [left, right, fused]
    overlay._branch_lineage_global[overlay._adapter_branch_key(fused)] = overlay._lineage_node_template(generation=2)
    overlay._branch_lineage_by_signature[sig] = {overlay._adapter_branch_key(fused): overlay._lineage_node_template(generation=2)}
    overlay._refresh_persistent_prior()
    prior = overlay._persistent_priors_by_signature[sig]
    assert prior['template_count'] == 3
    assert prior['memory_horizon'] >= 4


def test_shadowhott_split_branch_uses_shorter_memory_horizon():
    cfg = GPTConfig(
        vocab_size=65, sequence_len=16, n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
        shadow_lineage_split_memory_scale=0.5,
        shadow_lineage_split_replay_decay_scale=0.8,
    )
    overlay = GPT(cfg).shadowhott
    sig = 'spl|truth|prov_hi|contr_clean'
    parent = {
        'candidate_id': 1,
        'context_signature': sig,
        'winning_mode': 'candidate_branch:parent',
        'attn_gates': torch.full((cfg.shadow_layers,), 1.10),
        'mlp_gates': torch.full((cfg.shadow_layers,), 1.10),
        'resid_gates': torch.full((cfg.shadow_layers,), 1.10),
        'logit_scale': 1.10,
        'accepted_score_delta': 0.2,
        'intervention_strength': 0.2,
        'persist_score': 0.5,
        'task_family': 'spl', 'regime_bucket': 'truth', 'provenance_bucket': 'prov_hi', 'contradiction_bucket': 'contr_clean',
    }
    split = dict(parent)
    split.update({
        'candidate_id': 2,
        'winning_mode': 'candidate_branch:split_left',
        'split_from': overlay._adapter_branch_key(parent),
        'memory_horizon': 2,
        'replay_decay': 0.76,
        'accepted_score_delta': 0.35,
    })
    overlay.accepted_templates = [parent, split]
    overlay.accepted_templates_by_signature[sig] = [parent, split]
    overlay._branch_lineage_global[overlay._adapter_branch_key(split)] = overlay._lineage_node_template(generation=0)
    overlay._branch_lineage_by_signature[sig] = {overlay._adapter_branch_key(split): overlay._lineage_node_template(generation=0)}
    overlay._refresh_persistent_prior()
    prior = overlay._persistent_priors_by_signature[sig]
    assert prior['memory_horizon'] == 2
    assert abs(prior['replay_decay'] - 0.76) < 1e-6
