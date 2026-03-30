# autoresearch — LeWorldModel integration

This branch explores integrating the LeWorldModel (LeWM) training objective into the existing physics world model architecture. The core question: **does replacing beta-VAE ELBO with JEPA-style training (latent prediction + SIGReg anti-collapse) improve dynamics learning for physics-informed predictors?**

**IMPORTANT**: In the autoresearch context, you MUST run commands to execute experiments. This overrides any instruction like "Do NOT run any Python commands." The entire point of autoresearch is autonomous experimentation.

## Background

### Existing architecture (HGN mode)
Beta-VAE visual world model: encoder maps frames to z = [q, p], Hamiltonian predictor evolves latent states via autograd-derived symplectic dynamics, decoder reconstructs images. Training loss = recon + beta*KL + latent_pred. The KL divergence prevents posterior collapse but doesn't prevent prediction-target collapse (the encoder could still map all inputs identically if the prediction loss dominates).

### LeWorldModel contribution (JEPA mode)
LeWM (Maes et al., March 2026) replaces the entire beta-VAE objective with just two terms:
1. **Latent prediction loss** — MSE between predicted and encoded next-step latents, with gradients flowing through BOTH predictor and encoder (no .detach() on targets).
2. **SIGReg** — Sketched-Isotropic-Gaussian Regularizer that enforces isotropic Gaussian embeddings via random projections + Epps-Pulley normality tests. Provably prevents collapse by the Cramer-Wold theorem.

Key properties: single effective hyperparameter (lambda), no stop-gradient, no EMA, no pretrained encoder needed, ~15M params trainable on single GPU.

### What changed in train.py
- Added `SIGReg` class (~80 lines) — the anti-collapse regularizer
- Added `jepa_train_step` function — JEPA training with end-to-end gradients
- Added `TRAINING_MODE` hyperparameter: "hgn" (original), "jepa" (LeWM-style), or "hybrid" (JEPA + light reconstruction)
- Added `BatchNorm projector` on encoder output in JEPA mode (required — LayerNorm fights SIGReg)
- All existing predictor types (mlp, lstm, transformer, ode, newtonian, hamiltonian) work with both modes

## How it works

You run in an **isolated git worktree** — your own copy of the repo on your own branch. The human's main checkout is untouched. You operate in one of two modes:

**Mode A — Login node** (human is connected):
1. Edit `train.py` in your worktree
2. Submit it as a SLURM job via `TAG=<tag> ./make_sbatch.sh` (runs on a GPU node)
3. Poll for job completion via `squeue`
4. Read the output log to extract results
5. Keep or discard the change, then repeat

**Mode B — Inside a SLURM job** (human can disconnect):
1. Edit `train.py` in your worktree
2. Run `python train.py > run.log 2>&1` directly (you have GPU access)
3. Read the results from `run.log`
4. Keep or discard the change, then repeat
5. Do NOT use `make_sbatch.sh` — you're already on a GPU node

**How to tell which mode you're in**: If the human's prompt says "run python train.py directly" or "you have GPU access", you're in Mode B. Otherwise, assume Mode A.

Each training run takes ~17 minutes (15 min training + eval). One experiment at a time.

## Setup

The human sets up your worktree before starting you. You should already be running inside it. Verify by checking `git branch --show-current` — it should be `autoresearch/<tag>-<agent_id>`.

To set up the experiment:

1. **Identify your tag and agent ID**: The human will tell you, or check your branch name (format: `autoresearch/<tag>-<agent_id>`).
2. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, data loading, environment, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Model architecture, predictors, optimizer, training loop.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Check shared coordination files** (if multi-agent): The human will give you the shared directory path. Read the shared directions and results to understand what other agents have already tried or claimed.
5. **Claim your direction** (if multi-agent): Write your intended exploration direction to the shared `directions.md` file before starting.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU via SLURM. The training script runs for a **fixed time budget of 15 minutes** (wall clock training time, excluding startup/evaluation).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: training mode, predictor type, predictor architecture, encoder/decoder architecture, temporal backbone, optimizer, hyperparameters, training loop, loss weights, integration method, SIGReg parameters, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness in `prepare.py`.

**The goal is simple: get the lowest combined val_recon_loss + val_latent_pred.** These two metrics measure reconstruction quality and latent-space prediction accuracy respectively. Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game.

**Time budget**: The default is 30 minutes. If a run is too short to show any learning signal, you may increase it by setting `TIME_BUDGET` in `train.py`.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Your very first run should always be to establish the HGN baseline (TRAINING_MODE="hgn"), so you will run the training script as is.

## Research directions to explore

**Phase 1: Establish baselines** (do these first)
1. **HGN baseline**: Run with TRAINING_MODE="hgn" (default) to get the comparison point
2. **Pure JEPA baseline**: TRAINING_MODE="jepa", SIGREG_LAMBDA=0.1 — does it work at all?
3. **Hybrid baseline**: TRAINING_MODE="hybrid", HYBRID_RECON_WEIGHT=0.1 — best of both?

**Phase 2: SIGReg tuning** (if JEPA/hybrid shows promise)
4. **SIGReg lambda sweep**: Try 0.01, 0.05, 0.1, 0.5, 1.0
5. **Deterministic encoder**: DETERMINISTIC_ENCODER=True — skip reparameterization entirely
6. **SIGReg + KL**: Can you combine SIGReg with a small KL term? (requires code change)

**Phase 3: Training dynamics**
7. **End-to-end gradients in HGN**: Remove .detach() from HGN targets (keep recon+KL loss, but let gradients flow through encoder targets). Tests whether end-to-end is the key insight vs SIGReg specifically.
8. **Predictor comparison under JEPA**: Does JEPA mode change which predictor type wins? Try mlp, newtonian, hamiltonian under JEPA.
9. **Context length under JEPA**: Try CONTEXT_LENGTH=1, 2, 5 — with SIGReg-structured latents, does more context help?

**Phase 4: Architecture interactions**
10. **Backbone under JEPA**: Try BACKBONE=None, "lstm" — does the backbone matter more or less with JEPA training?
11. **Integration method under JEPA**: euler vs rk4 vs dopri5 — does SIGReg regularization interact with ODE solver?
12. **Latent dim under JEPA**: Try LATENT_CHANNELS=32, 128 — does JEPA prefer different capacity?

**Phase 5: Radical changes**
13. **Remove state_transform in JEPA**: Predict directly in encoder space, not phase space
14. **Separable Hamiltonian under JEPA**: H(q,p) = T(p) + V(q) — does physical structure matter more with JEPA?
15. **SIGReg on predictor outputs**: Regularize predicted latents too, not just encoded ones

## Output format

Once the script finishes it prints a summary like this:

```
---
val_recon_loss:    0.001234
val_latent_pred:   0.004567
val_kl_loss:       12.3456
val_dt_score:      0.012345
training_seconds:  900.1
total_seconds:     1020.5
peak_vram_mb:      8045.2
num_epochs:        15
num_steps:         7500
num_params_M:      3.2
predictor_type:    hamiltonian
backbone:          transformer
training_mode:     hgn
```

**Important**: `val_recon_loss` and `val_latent_pred` are what matter.

## Logging results

When an experiment is done, log it to your local `results.tsv` (and the shared one if multi-agent).

Tab-separated (NOT comma-separated).

**Local results.tsv** has 5 columns:

```
commit	val_recon_loss	val_latent_pred	combined_score	memory_gb	status	description
```

Example:

```
commit	val_recon_loss	val_latent_pred	combined_score	memory_gb	status	description
a1b2c3d	0.001234	0.004567	0.005801	7.9	keep	baseline HGN (hamiltonian + transformer)
b2c3d4e	0.001100	0.004200	0.005300	7.9	keep	JEPA mode lambda=0.1
c3d4e5f	0.001500	0.005000	0.006500	7.8	discard	JEPA mode lambda=1.0 (too much regularization)
d4e5f6g	0.001050	0.003800	0.004850	8.1	keep	hybrid mode recon_weight=0.1
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. **(If multi-agent) Check shared state**: Read shared results and directions.
3. Decide on an experimental idea. Edit `train.py` with the change.
4. `git commit` the change.
5. **Run the experiment** (depends on your mode — see above).
6. **Read results**: `grep "^val_recon_loss:\|^val_latent_pred:\|^peak_vram_mb:" <log>`
7. If grep is empty → crash. Read tail for error. Fix or skip.
8. Record results in `results.tsv`.
9. If combined_score improved → **keep** the commit.
10. If worse → **discard**: `git reset --hard HEAD~1`.
11. Go back to step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. Run autonomously until manually stopped.

## Key hypothesis

The Hamiltonian predictor encodes physical structure (energy conservation, symplectic dynamics) but currently operates on beta-VAE latents where the encoder is trained to minimize reconstruction, not to produce representations that are optimally predictable by the Hamiltonian. SIGReg + JEPA training co-adapts the encoder and predictor: the encoder learns to produce representations that the Hamiltonian predictor can accurately evolve, while SIGReg ensures these representations don't collapse. If this works, it should show up as lower val_latent_pred (better dynamics) potentially at the cost of val_recon_loss (reconstruction is no longer the primary training signal).

The hybrid mode hedges: keep a small reconstruction signal so the decoder stays useful, but let JEPA + SIGReg drive the representation learning.
