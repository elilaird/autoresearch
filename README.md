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
cd autoresearch
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

## Running the agent

Spin up Claude Code (or any coding agent) in this directory, then prompt:

```
Read program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will:
1. Create a branch `autoresearch/<tag>`
2. Run the baseline to establish the starting `val_dt_score`
3. Begin the autonomous experiment loop — modifying `train.py`, running, evaluating, keeping or discarding

Each experiment takes ~17 minutes. The agent runs indefinitely until you stop it.

### On a SLURM cluster

Submit as a SLURM job with enough time for many experiments:

```bash
# Example: 8 hours = ~30 experiments
GPU=1 CPUS=16 MEM=96G TIME=0-08:00:00 PARTITION=batch \
    CONDA_ENV=world_models ./make_sbatch.sh \
    cd autoresearch && claude --print "read program.md and start experimenting"
```

## Results

Experiments are logged to `results.tsv` (tab-separated, untracked by git):

```
commit	val_dt_score	memory_gb	status	description
a1b2c3d	0.012345	7.9	keep	baseline (hamiltonian + transformer)
b2c3d4e	0.011200	7.9	keep	increase beta to 0.01
c3d4e5f	0.013000	7.8	discard	switch to MLP predictor
```

### Analysis

Open `analysis.ipynb` to visualize experiment progress:

```bash
jupyter notebook analysis.ipynb
```

This generates `progress.png` showing the improvement frontier over time.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Fixed time budget (15 min).** Experiments are directly comparable regardless of architecture changes. ODE predictors get fewer steps but the same wall-clock time.
- **dt generalization metric.** Unlike reconstruction loss, this measures what matters: can the model predict dynamics at rates it wasn't trained on?
- **Physics-informed starting point.** The baseline uses a Hamiltonian predictor because the research question is about physics-informed architectures. The agent can try simpler predictors to establish baselines.
- **Self-contained.** No Hydra configs, no wandb, no distributed training. One GPU, one file, one metric.

## Project structure

```
prepare.py      — data loading, environment, evaluation harness (read-only)
train.py        — model architecture, predictors, training loop (agent modifies)
program.md      — agent instructions (human modifies)
results.tsv     — experiment log (untracked)
run.log         — latest run output (untracked)
analysis.ipynb  — visualization notebook
pyproject.toml  — dependencies (informational)
```
