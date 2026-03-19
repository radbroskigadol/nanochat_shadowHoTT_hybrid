# \# nanochat-shadowhott-hybrid

# 

# !\[nanochat logo](dev/nanochat.png)

# 

# A nanochat-derived research fork for experimenting with \*\*ShadowHoTT overlays\*\*, \*\*bounded late-layer intervention\*\*, and \*\*hybrid optimizer behavior\*\* inside a small, hackable LLM training stack.

# 

# This repository keeps the basic spirit of the original nanochat codebase: minimal moving parts, readable PyTorch, single-node friendliness, and end-to-end coverage from tokenizer training through pretraining, chat finetuning, evaluation, inference, and a web UI. The difference is that this fork is not just a clean baseline. It introduces an additional \*\*meta-control layer\*\* that observes coarse runtime state, maintains a \*\*four-valued ShadowHoTT-style semantic state\*\*, emits \*\*bounded modulation signals\*\* for late transformer layers, and can \*\*accept, reject, retain, and route intervention templates\*\* by context-sensitive signatures.

# 

# \## Status

# 

# This is an \*\*experimental research fork\*\*, not a drop-in replacement for upstream nanochat.

# 

# It is currently best understood as:

# 

# \- a runnable nanochat-based LLM harness

# \- plus a first working \*\*ShadowHoTT overlay\*\*

# \- plus benchmark-driven \*\*keep/revert persistence logic\*\*

# \- plus branch/lineage-style retained adapter search

# \- plus small infrastructure changes for dtype control, Flash Attention fallback, and testing

# 

# Some upstream nanochat claims, benchmarks, cost figures, and leaderboard entries describe the upstream project rather than this fork unless explicitly re-measured here.

# 

# \## What this fork adds

# 

# \### ShadowHoTT overlay

# 

# This fork adds `nanochat/shadowhott.py` and wires it into the GPT model in `nanochat/gpt.py`.

# 

# The overlay currently provides:

# 

# \- a \*\*four-valued shadow state\*\*: `T`, `F`, `B`, `N`

# \- bilateral score tracking via \*\*truth mass\*\* and \*\*falsity mass`

# \- a small \*\*meta-transformer controller\*\* operating on coarse runtime metadata

# \- \*\*bounded late-layer intervention gates\*\* applied to selected upper transformer layers

# \- deterministic \*\*shadow reports\*\* with a certification sidecar

# \- helper methods to inspect and reset runtime shadow state

# 

# Conceptually, the base transformer remains intact. The ShadowHoTT layer does not replace the model's main forward pass. Instead, it acts as a second-order control/readout mechanism over it.

# 

# \### Persistence and acceptance flow

# 

# This fork also adds a persistence harness around ShadowHoTT interventions.

# 

# Accepted intervention candidates can be retained as bounded gate or logit templates and blended back into later passes as priors. The model exposes methods for:

# 

# \- reading the current acceptance summary

# \- manually accepting the last intervention candidate

# \- manually rejecting the last intervention candidate

# 

# This makes the overlay more than a passive monitor: it can accumulate a lightweight history of which intervention patterns helped or hurt.

# 

# \### Automatic benchmark-driven keep/revert

# 

# The model includes `shadow\_benchmark\_batch(...)`, which compares a ShadowHoTT-controlled pass against a baseline pass on the same batch and decides whether the pending candidate should be accepted or rejected.

# 

# `scripts/base\_train.py` exposes corresponding training flags such as:

# 

# \- `--shadow-auto-eval-every`

# \- `--shadow-accept-threshold`

# \- `--shadow-reject-threshold`

# 

# This allows the overlay to operate in either:

# 

# \- \*\*observation-only mode\*\*, where reports are generated without retention, or

# \- \*\*benchmark-driven mode\*\*, where retained intervention templates are updated during training

# 

# \### Context-sensitive routing and lineage behavior

# 

# Retained intervention templates are keyed by richer runtime signatures rather than only crude pre-pass routing labels.

# 

# This fork includes machinery for:

# 

# \- distinguishing \*\*pre-pass context\*\* from \*\*live post-pass regime\*\*

# \- tracking semantic drift between those regimes

# \- signature-sensitive retained prior blending

# \- branch-style candidate generation and adaptation

# \- lineage fusion and lineage splitting behavior for retained branches

# 

# At the moment this is still experimental, but it already pushes the fork beyond a static overlay into a simple regime-conditioned adaptation system.

# 

# \### Runtime and infrastructure changes

# 

# Other notable changes include:

# 

# \- explicit global compute dtype control through `NANOCHAT\_DTYPE`

# \- storage of master weights in fp32 with forward compute cast by custom linear layers

# \- Flash Attention integration with \*\*PyTorch SDPA fallback\*\* where FA3 is unavailable

# \- additional tests for fallback attention and ShadowHoTT behavior

# 

# \## Project philosophy

# 

# The goal of this fork is not to turn nanochat into a giant framework. The appeal of the base code remains the same: it is compact, readable, and easy to modify. The purpose of this branch is to test whether a \*\*semantic-control overlay\*\* can be layered on top of a relatively standard transformer stack without destroying the simplicity that makes the codebase useful.

# 

# Stated more bluntly:

# 

# \- the transformer should still be understandable

# \- the extra machinery should still be inspectable

# \- new behavior should be added in a way that can actually be trained and tested

# 

# \## Repository layout

# 

# ```text

# .

# ├── LICENSE

# ├── README.md

# ├── SHADOWHOTT\_CHANGES.md

# ├── dev

# │   ├── LEADERBOARD.md

# │   ├── LOG.md

# │   ├── estimate\_gpt3\_core.ipynb

# │   ├── gen\_synthetic\_data.py

# │   ├── generate\_logo.html

# │   ├── nanochat.png

# │   ├── repackage\_data\_reference.py

# │   └── scaling\_laws\_jan26.png

# ├── nanochat

# │   ├── checkpoint\_manager.py

# │   ├── common.py

# │   ├── core\_eval.py

# │   ├── dataloader.py

# │   ├── dataset.py

# │   ├── engine.py

# │   ├── execution.py

# │   ├── flash\_attention.py

# │   ├── fp8.py

# │   ├── gpt.py

# │   ├── loss\_eval.py

# │   ├── optim.py

# │   ├── report.py

# │   ├── shadowhott.py

# │   ├── tokenizer.py

# │   └── ui.html

# ├── runs

# │   ├── miniseries.sh

# │   ├── runcpu.sh

# │   ├── scaling\_laws.sh

# │   └── speedrun.sh

# ├── scripts

# │   ├── base\_eval.py

# │   ├── base\_train.py

# │   ├── chat\_cli.py

# │   ├── chat\_eval.py

# │   ├── chat\_rl.py

# │   ├── chat\_sft.py

# │   ├── chat\_web.py

# │   ├── tok\_eval.py

# │   └── tok\_train.py

# ├── tasks

# │   ├── arc.py

# │   ├── common.py

# │   ├── customjson.py

# │   ├── gsm8k.py

# │   ├── humaneval.py

# │   ├── mmlu.py

# │   ├── smoltalk.py

# │   └── spellingbee.py

# ├── tests

# │   ├── test\_attention\_fallback.py

# │   ├── test\_engine.py

# │   └── test\_shadowhott\_overlay.py

# ├── pyproject.toml

# └── uv.lock

# ```

# 

# \## Installation

# 

# This repo uses `uv` for environment management.

# 

# \### CPU environment

# 

# ```bash

# uv venv

# uv sync --extra cpu

# source .venv/bin/activate

# ```

# 

# \### GPU environment

# 

# ```bash

# uv venv

# uv sync --extra gpu

# source .venv/bin/activate

# ```

# 

# The project is configured to target either CPU wheels or CUDA 12.8 wheels for PyTorch through `uv` sources in `pyproject.toml`.

# 

# \## Quick start

# 

# \### 1. Train tokenizer

# 

# ```bash

# python -m nanochat.dataset -n 8

# python -m scripts.tok\_train

# python -m scripts.tok\_eval

# ```

# 

# \### 2. Run base training

# 

# For a small local experiment:

# 

# ```bash

# python -m scripts.base\_train \\

# &#x20;   --depth=6 \\

# &#x20;   --head-dim=64 \\

# &#x20;   --window-pattern=L \\

# &#x20;   --max-seq-len=512 \\

# &#x20;   --device-batch-size=32 \\

# &#x20;   --total-batch-size=16384 \\

# &#x20;   --eval-every=100 \\

# &#x20;   --eval-tokens=524288 \\

# &#x20;   --core-metric-every=-1 \\

# &#x20;   --sample-every=100 \\

# &#x20;   --num-iterations=5000 \\

# &#x20;   --run=local\_test

# ```

# 

# For distributed GPU training, use `torchrun` in the usual way. The provided scripts in `runs/` are the best place to start.

# 

# \### 3. Evaluate

# 

# ```bash

# python -m scripts.base\_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# ```

# 

# \### 4. Finetune for chat

# 

# ```bash

# python -m scripts.chat\_sft \\

# &#x20;   --max-seq-len=512 \\

# &#x20;   --device-batch-size=32 \\

# &#x20;   --total-batch-size=16384 \\

# &#x20;   --eval-every=200 \\

# &#x20;   --eval-tokens=524288 \\

# &#x20;   --num-iterations=1500 \\

# &#x20;   --run=chat\_sft

# ```

# 

# \### 5. Talk to the model

# 

# CLI:

# 

# ```bash

# python -m scripts.chat\_cli -p "hello"

# ```

# 

# Web UI:

# 

# ```bash

# python -m scripts.chat\_web

# ```

# 

# Then open the displayed local URL in your browser.

# 

# \## ShadowHoTT-specific training controls

# 

# The base training script exposes a large number of ShadowHoTT-related flags. The most important first ones to know are:

# 

# ```bash

# \--shadow-auto-eval-every

# \--shadow-accept-threshold

# \--shadow-reject-threshold

# \--shadow-loss-weight

# \--shadow-regime-weight

# \--shadow-drift-weight

# \--shadow-sparsity-weight

# \--shadow-provenance-weight

# \--shadow-consistency-weight

# ```

# 

# These let you decide whether the overlay is merely reporting, lightly regularizing, or actively performing benchmark-driven keep/revert decisions during training.

# 

# There are also more experimental controls for retained adapter branching, inner-loop adaptation, and lineage fusion/splitting. Those are intended for research use, not for claiming stable best practices yet.

# 

# \## CPU / MPS usage

# 

# `runs/runcpu.sh` demonstrates a much smaller local run intended for CPU or Apple Silicon experimentation.

# 

# This path is mainly educational. It is useful for checking code paths, testing the training loop, and exercising the stack without a large GPU machine, but you should not expect strong capability from these runs.

# 

# \## Precision and dtype behavior

# 

# This fork does not rely on `torch.amp.autocast` as the primary precision-management mechanism.

# 

# Instead:

# 

# \- master weights stay in fp32 for optimizer precision

# \- compute dtype is controlled globally through `COMPUTE\_DTYPE`

# \- custom linear layers cast weights to the active compute dtype during forward

# \- embeddings are stored directly in the compute dtype to save memory

# 

# Default behavior is hardware-sensitive:

# 

# \- Ampere/Hopper-class CUDA devices default to `bfloat16`

# \- older CUDA devices default to `float32`

# \- CPU and MPS default to `float32`

# 

# You can override this with:

# 

# ```bash

# NANOCHAT\_DTYPE=float32 python -m scripts.chat\_cli -p "hello"

# NANOCHAT\_DTYPE=bfloat16 torchrun --nproc\_per\_node=8 -m scripts.base\_train

# ```

# 

# \## Attention backend behavior

# 

# The codebase includes a custom Flash Attention wrapper in `nanochat/flash\_attention.py`.

# 

# Where supported, it uses Flash Attention style kernels. Where that path is unavailable, it falls back to PyTorch SDPA. This keeps the code portable while still allowing newer hardware to benefit from the faster path.

# 

# \## Testing

# 

# Run the tests with:

# 

# ```bash

# pytest

# ```

# 

# The tests currently include coverage for:

# 

# \- engine behavior

# \- attention fallback behavior

# \- ShadowHoTT overlay forward pass and report generation

# \- acceptance/rejection persistence cycle

# \- automatic benchmark-driven accept/reject

# \- context-sensitive retained prior routing

# \- signature reporting and semantic drift behavior

# 

# \## Notes for public GitHub publication

# 

# If you are publishing this repository publicly, make the presentation honest:

# 

# \- describe it as a \*\*nanochat-derived fork\*\*

# \- do not imply that upstream nanochat leaderboard or cost claims were reproduced by this branch unless you actually reran them here

# \- keep upstream credit and the original license

# \- clearly separate upstream nanochat facts from ShadowHoTT-specific additions in this branch

# 

# \## Upstream attribution

# 

# This project is derived from \*\*Andrej Karpathy's nanochat\*\* and retains the original MIT licensing structure.

# 

# nanochat itself was positioned as a minimal full-stack LLM training and chat stack. This fork keeps that codebase as its foundation while adding the ShadowHoTT overlay and related experimental control machinery.

# 

# \## Citation

# 

# If you want to cite this fork, use a fork-specific entry such as:

# 

# ```bibtex

# @misc{nanochat\_shadowhott\_hybrid,

# &#x20; author = {David Betzer},

# &#x20; title = {nanochat-shadowhott-hybrid: a nanochat-derived ShadowHoTT research fork},

# &#x20; year = {2026},

# &#x20; publisher = {GitHub},

# &#x20; note = {Fork of Andrej Karpathy's nanochat with ShadowHoTT overlay and hybrid optimizer experiments}

# }

# ```

# 

# If your work depends materially on upstream nanochat as well, cite the upstream repository separately.

# 

# \## License

# 

# MIT.

# 

# See `LICENSE`.



