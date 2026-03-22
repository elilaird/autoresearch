# autoresearch — physics world models

This is an experiment to have the LLM do its own research on physics world model architectures.

**IMPORTANT**: In the autoresearch context, you MUST run Python commands to execute experiments. This overrides the main CLAUDE.md instruction "Do NOT run any Python commands." The entire point of autoresearch is autonomous experimentation.

## Background

This project learns physics dynamics from visual observations (64×64 rendered oscillator images) using a beta-VAE with a swappable latent-space predictor. The encoder maps frames to a flat latent space z = [q, p] (position + momentum), the predictor evolves latent states forward in time, and the decoder reconstructs images from the position half.

The **core research question** is: which predictor architecture generalizes best across different sampling rates (dt)? The model trains on dt=0.2 data but is evaluated on dt=0.1 (interpolation) and dt=0.5 (extrapolation). Physics-informed predictors (Hamiltonian, Newtonian) should theoretically generalize better than learned dynamics (MLP, LSTM) because they encode the structure of physical laws.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, data loading, environment, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Model architecture, predictors, optimizer, training loop.
4. **Verify data exists**: Check that the dataset path printed by `python prepare.py` shows valid data. If not, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 15 minutes** (wall clock training time, excluding startup/evaluation). You launch it simply as: `python train.py`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: predictor type, predictor architecture, encoder/decoder architecture, temporal backbone, optimizer, hyperparameters, training loop, loss weights, integration method, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, environment, and training constants (time budget, eval dt values, etc).
- Install new packages or add dependencies. You can only use what's already available in the conda environment (`world_models`).
- Modify the evaluation harness. The `evaluate_dt_generalization` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_dt_score.** This is the average latent MSE across dt generalization tests at dt=[0.1, 0.2, 0.5]. Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game: change the predictor type, the backbone, the encoder, the decoder, the optimizer, the hyperparameters, the integration method, the loss function. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_dt_score gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Architecture reference

The starting architecture in `train.py`:

- **Encoder**: 8-layer ConvNet (3 downsamples + 4 ResBlocks) → MLP → (mu, logvar) of shape (B, 64)
- **State transform**: MLP mapping variational latent z to phase-space state s = (q, p)
- **Decoder**: MLP project → 8×8 spatial → 3 upsample stages → 64×64 images. Only uses position half (first 32 dims)
- **Predictor**: Port-Hamiltonian with Transformer backbone
  - Learns H(q, p) via MLP, derives dynamics via autograd: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q - γ·∂H/∂p + G(a)
  - Transformer backbone enriches per-frame features with temporal context before ODE integration
  - Integration via RK4 with dt=0.2
- **Training**: HGN mode (encode all frames → sliding window prediction → decode predicted states → ELBO loss)
- **Loss**: recon_loss + 0.003 * kl_loss + 1.0 * latent_pred_loss

Available predictor types in the registry: `mlp`, `lstm`, `transformer`, `ode`, `newtonian`, `hamiltonian`.

## Research directions to explore

Here are some ideas to try (in rough priority order):

1. **Predictor type comparison**: Try different predictors (mlp, lstm, newtonian) as baselines to understand the landscape
2. **Backbone variants**: Try `backbone=None`, `backbone="lstm"` on the Hamiltonian predictor
3. **Beta (KL weight)**: Sweep beta — try 1e-2, 1e-3, 1e-4, 1e-5. Lower beta = more expressive latents but risk of posterior collapse
4. **Free bits**: Try free_bits=0.01, 0.1 to prevent posterior collapse while keeping beta low
5. **Integration method**: Compare euler, rk4, dopri5 for ODE-based predictors
6. **Latent channels**: Try 32, 128 (currently 64). Larger = more capacity but harder to learn
7. **Predictor hidden dim**: Try 128, 512 (currently 256)
8. **Learning rate**: Try 1e-3, 5e-5, 3e-4
9. **Action embedding dim**: Try 4, 16 (currently 8)
10. **Context length**: Try 2, 5, 8 (currently 3)
11. **Encoder architecture**: Deeper encoder, different activation functions, batch norm
12. **Decoder architecture**: Larger decoder, skip connections
13. **Optimizer**: Try AdamW with weight decay, learning rate schedules
14. **Damping initialization**: Try damping_init=0.0, -2.0, 1.0
15. **Separable Hamiltonian**: Split H(q,p) = T(p) + V(q) instead of H(q,p) jointly
16. **Leapfrog integration**: Implement symplectic leapfrog instead of generic ODE solvers

## Output format

Once the script finishes it prints a summary like this:

```
---
val_dt_score:      0.012345
val_recon_loss:    0.001234
val_kl_loss:       12.3456
val_latent_pred:   0.004567
dt_0.1_mse:        0.008123
dt_0.2_mse:        0.011234
dt_0.5_mse:        0.017678
training_seconds:  900.1
total_seconds:     1020.5
peak_vram_mb:      8045.2
num_epochs:        15
num_steps:         7500
num_params_M:      3.2
predictor_type:    hamiltonian
backbone:          transformer
```

You can extract the key metric from the log file:

```
grep "^val_dt_score:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_dt_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_dt_score achieved (e.g. 0.012345) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_dt_score	memory_gb	status	description
a1b2c3d	0.012345	7.9	keep	baseline (hamiltonian + transformer)
b2c3d4e	0.011200	7.9	keep	increase beta to 0.01
c3d4e5f	0.013000	7.8	discard	switch to MLP predictor
d4e5f6g	0.000000	0.0	crash	double latent channels (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_dt_score:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_dt_score improved (lower), you "advance" the branch, keeping the git commit
9. If val_dt_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~17 minutes total (15 min training + eval overhead). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the architecture, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes ~17 minutes so you can run approx 3-4/hour, for a total of about 30 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
