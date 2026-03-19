# ShadowHoTT overlay changes

This branch adds a first runnable ShadowHoTT overlay to nanochat without replacing the base transformer.

## Added
- `nanochat/shadowhott.py`
  - four-valued shadow state (`T`, `F`, `B`, `N`)
  - bilateral score layer (`truth_mass`, `falsity_mass`)
  - tiny meta-transformer controller
  - bounded late-layer intervention gates
  - certified invariant / determinized reports
- `tests/test_shadowhott_overlay.py`
  - smoke test for forward pass and report generation

## Modified
- `nanochat/gpt.py`
  - ShadowHoTT config fields on `GPTConfig`
  - `Block.forward(...)` now supports bounded shadow gates
  - `GPT` now builds the overlay, applies late-layer intervention, keeps a shadow report, and includes shadow params in optimizer groups
  - `reset_shadowhott_state()` and `get_shadow_report()` helper methods

## What this first version does
- keeps the base transformer intact
- adds a second-layer transformer that reads coarse runtime metadata
- emits bounded gates for the top shadow layers
- maintains a four-valued / bilateral shadow state
- produces a deterministic shadow report with a certification sidecar

## What it does not do yet
- persistent adapter banks
- proof-checked rewrite acceptance
- retrieval/provenance integration beyond local report fields
- long-horizon recursive optimizer loops


## Persistence harness (v1)

- Added keep/revert acceptance flow for the last ShadowHoTT intervention candidate.
- Accepted interventions are stored as bounded gate/logit templates.
- Accepted templates are blended back into later runs as a persistent prior.
- New model methods:
  - `get_shadow_acceptance_summary()`
  - `shadow_accept_last(score_delta, note="")`
  - `shadow_reject_last(score_delta, note="")`
- Certificate sidecar now reports candidate / accepted / rejected status.


## Automatic benchmark-driven persistence
- added `GPT.shadow_benchmark_batch(...)` to compare a shadow-controlled pass against a baseline pass on the same batch
- the method auto-accepts or auto-rejects the pending ShadowHoTT intervention template by measured loss delta
- `scripts/base_train.py` now supports `--shadow-auto-eval-every`, `--shadow-accept-threshold`, and `--shadow-reject-threshold` for benchmark-driven keep/revert during training
- reports can run in observation-only mode without creating a persistence candidate

- persistence signatures now distinguish **pre-pass** routing context from **live post-pass** semantic regime
- accepted/rejected intervention templates are keyed by the **live** post-pass regime signature derived from current four-valued histogram, provenance coherence, and contradiction pressure
- reports now include `semantic_drift`, `pre_context_signature`, and `pre_context_signature_components`
- acceptance summaries now include `pending_pre_context_signature`

- Added lineage fusion/splitting pass: mature compatible branches can emit fused candidates, while unstable newborn branches can emit split children with boosted mutation radius.
