# autoresearch — physics world models

Autonomous AI-driven architecture search for physics world models. Based on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern, adapted for visual dynamics learning with Hamiltonian generative networks.

The idea: give an AI agent a visual world model codebase and let it experiment autonomously overnight. It modifies the architecture, trains for 15 minutes, checks if dt generalization improved, keeps or discards, and repeats. You wake up to a log of ~30 experiments and (hopefully) a better model.

## How it works

Three files matter:

- **`prepare.py`** — fixed evaluation harness: data loading, oscillator environment + rendering, rollout evaluation, and the ground-truth metric (`val_dt_score`). **Read-only — do not modify.**
- **`train.py`** — the single file the agent edits. Contains the full visual world model: encoder, decoder, 6 predictor types (MLP, LSTM, Transformer, ODE, Newtonian, Hamiltonian), temporal backbones, optimizer, and training loop. **This file is edited by the agent.**
- **`program.md`** — instructions for the agent. Describes the experiment loop, what can/cannot be modified, and research directions. **This file is edited by the human.**

### The metric

**`val_dt_score`** = average latent MSE across dt generalization tests at dt=[0.1, 0.2, 0.5]. Lower is better.

The model trains on dt=0.2 oscillator data. At evaluation, fresh trajectories are generated at three different sampling rates, and the model must predict dynamics accurately across all of them. Physics-informed predictors (Hamiltonian, Newtonian) should generalize better because they encode the structure of physical laws.

### The architecture

The starting point is a beta-VAE visual world model:

1. **Encoder**: 8-layer ConvNet → flat latent z = [q, p] (position + momentum)
2. **State transform**: MLP mapping variational latent to phase-space state
3. **Predictor**: Port-Hamiltonian with Transformer backbone — learns H(q,p), derives dynamics via autograd with symplectic structure
4. **Decoder**: Reconstructs 64×64 images from position half of latent

Training uses the HGN (Hamiltonian Generative Network) approach: encode all frames → sliding window prediction → decode predicted states → ELBO loss.

## Quick start

**Requirements:** Single NVIDIA GPU, Python 3.10+, conda environment `world_models` with PyTorch and torchdiffeq.

```bash
# 1. Activate the conda environment
conda activate world_models

# 2. Verify data exists (checks dataset on lustre, prints stats)
python prepare.py

# 3. Run a single training experiment (~17 min: 15 min training + eval)
python train.py
```

If `prepare.py` finds the data and `train.py` completes with a `val_dt_score` printed, your setup is working.

### Custom data path

If your datasets are not on lustre, set the `DATA_ROOT` environment variable:

```bash
export DATA_ROOT=/path/to/your/datasets
python prepare.py  # should find oscillator_visual_60k_dt20Hz/
```

## Running agents

Each agent runs in its own **git worktree** — an isolated copy of the repo on its own branch. Your main checkout is never touched. A single script `run_agent.sh` handles worktree creation and launch.

#### Option A: Interactive on login node (requires active session)

```bash
TAG=mar23 AGENT=0 ./run_agent.sh
```

Creates worktree at `~/Projects/autoresearch-agents/mar23-0/`, then launches Claude Code interactively. Claude submits SLURM jobs for each training run and polls for results.

**To stop:** `Ctrl+C`, then `scancel <job_id>` if a SLURM job is still running.

#### Option B: SLURM job (survives disconnect)

```bash
TAG=mar23 AGENT=1 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
```

Creates worktree, then submits a SLURM job where Claude runs with direct GPU access. You can log off — it keeps running.

**To stop:** `scancel <job_id>`

**To monitor:**
```bash
# SLURM output
tail -f ~/Projects/autoresearch-agents/mar23-1/output/train/agent_*.out

# Shared results (all agents)
cat /lustre/.../autoresearch/shared/mar23/results.tsv

# Wandb — runs are logged to the "autoresearch" project
```

#### Mixing modes

Run some agents interactively and others in SLURM — all share the same coordination files:

```bash
# Agent 0: interactive
TAG=mar23 AGENT=0 ./run_agent.sh

# Agent 1 & 2: fire-and-forget SLURM jobs
TAG=mar23 AGENT=1 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
TAG=mar23 AGENT=2 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
```

All agents share `results.tsv` and `directions.md` on lustre, so they coordinate to avoid duplicating experiments.

## Results

Experiments are logged to `results.tsv` (tab-separated, untracked by git):

```
commit	val_dt_score	memory_gb	status	description
a1b2c3d	0.012345	7.9	keep	baseline (hamiltonian + transformer)
b2c3d4e	0.011200	7.9	keep	increase beta to 0.01
c3d4e5f	0.013000	7.8	discard	switch to MLP predictor
```

The shared file on lustre has an extra `agent` column so you can see which agent tried what.

### Analysis

Open `analysis.ipynb` to visualize experiment progress:

```bash
jupyter notebook analysis.ipynb
```

This generates `progress.png` showing the improvement frontier over time.

### Wandb

Training runs are logged to the `autoresearch` wandb project (if wandb is installed). Each run tracks:
- Per-step: loss components (recon, KL, latent pred)
- Final: val_dt_score, per-dt MSE breakdown, VRAM usage
- Config: all hyperparameters, git hash, branch name

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget (15 min).** Experiments are directly comparable regardless of architecture changes. ODE predictors get fewer steps but the same wall-clock time.
- **dt generalization metric.** Unlike reconstruction loss, this measures what matters: can the model predict dynamics at rates it wasn't trained on?
- **Physics-informed starting point.** The baseline uses a Hamiltonian predictor because the research question is about physics-informed architectures. The agent can try simpler predictors to establish baselines.
- **Git worktrees for isolation.** Each agent gets its own working directory and branch. No conflicts between agents or with the human's main checkout.
- **Shared coordination files.** Agents read a shared results log before each experiment to avoid duplicating work. Advisory, not enforced — simple and robust.

## Project structure

```
prepare.py        — data loading, environment, evaluation harness (read-only)
train.py          — model architecture, predictors, training loop (agent modifies)
program.md        — agent instructions (human modifies)
run_agent.sh      — create worktree + launch agent (interactive or SLURM)
results.tsv       — experiment log (untracked)
run.log           — latest run output (untracked)
analysis.ipynb    — visualization notebook
pyproject.toml    — dependencies (informational)
```
