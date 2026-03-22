"""
Fixed evaluation harness for autoresearch physics world model experiments.

This file is READ-ONLY. Do not modify it.
It provides data loading, environment simulation, rollout evaluation,
and the fixed metric (val_dt_score) used to compare experiments.

Usage:
    python prepare.py           # verify data exists and print stats
"""

import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 900  # training time budget in seconds (15 minutes)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    "/lustre/smuexa01/client/users/ejlaird/physics-world-models/datasets",
)
DATASET_NAME = "oscillator_visual_60k_dt20Hz"
DATASET_VERSION = "2026-02-20_13-09-05"
DATASET_PATH = os.path.join(DATA_ROOT, DATASET_NAME, DATASET_VERSION)

# Evaluation
EVAL_DT_VALUES = [0.1, 0.2, 0.5]
EVAL_N_SEQS = 16
EVAL_SEQ_LEN = 30
EVAL_SEED = 12345  # fixed seed for reproducible evaluation

# Environment
IMG_SIZE = 64
ACTION_DIM = 3
STATE_DIM = 2

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class PrecomputedDataset(Dataset):
    """Dataset that loads memory-mapped numpy arrays for efficient chunk loading."""

    def __init__(self, path):
        if path.endswith(".pt"):
            data = torch.load(path, weights_only=False)
            self.states = data["states"]
            self.actions = data["actions"]
            self.images = data.get("images")
            self.mmap = False
        else:
            if not path.endswith(".npz"):
                path = path + ".npz" if not os.path.exists(path) else path
            data = np.load(path, mmap_mode="r")
            self.states = data["states"]
            self.actions = data["actions"]
            self.images = data.get("images")
            self.mmap = True

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        states = (
            torch.from_numpy(np.array(self.states[idx]))
            if self.mmap
            else self.states[idx]
        )
        actions = (
            torch.from_numpy(np.array(self.actions[idx]))
            if self.mmap
            else self.actions[idx]
        )

        item = {"states": states, "actions": actions}

        if self.images is not None:
            img = self.images[idx]
            if self.mmap:
                img = np.array(img)
                img = torch.from_numpy(img)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            item["images"] = img

        return item


def make_dataloaders(batch_size):
    """Create train and val DataLoaders from the fixed dataset.

    Returns:
        (train_loader, val_loader) tuple of DataLoaders.
    """
    train_path = os.path.join(DATASET_PATH, "train.npz")
    val_path = os.path.join(DATASET_PATH, "val.npz")

    train_data = PrecomputedDataset(train_path)
    val_data = PrecomputedDataset(val_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def batch_to_device(batch, device):
    """Move a batch dict to the given device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------

DEFAULT_BG_COLOR = [81.0 / 255, 88.0 / 255, 93.0 / 255]
DEFAULT_BALL_COLORS = [
    [173.0 / 255, 146.0 / 255, 0.0],
    [173.0 / 255, 0.0, 0.0],
    [0.0, 146.0 / 255, 0.0],
]


def world_to_pixels(x, y, res, world_size):
    """Map world coordinates to pixel coordinates."""
    if isinstance(x, torch.Tensor):
        px = int((res * (x + world_size) / (2 * world_size)).long().item())
    else:
        px = int(res * (x + world_size) / (2 * world_size))
    if isinstance(y, torch.Tensor):
        py = int((res * (y + world_size) / (2 * world_size)).long().item())
    else:
        py = int(res * (y + world_size) / (2 * world_size))
    return px, py


def render_circle_aa(img, center_x, center_y, radius, color, render_quality="medium"):
    """Render an anti-aliased circle using a distance field."""
    H, W, C = img.shape
    device = img.device

    center_x_val = float(center_x)
    center_y_val = float(center_y)

    scale_factor = {"low": 1, "medium": 2, "high": 4}[render_quality]
    H_scaled, W_scaled = H * scale_factor, W * scale_factor

    center_x_scaled = center_x_val * scale_factor
    center_y_scaled = center_y_val * scale_factor
    radius_scaled = radius * scale_factor

    y_coords = torch.arange(H_scaled, dtype=torch.float32, device=device) + 0.5
    x_coords = torch.arange(W_scaled, dtype=torch.float32, device=device) + 0.5
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    dist = torch.sqrt((X - center_x_scaled) ** 2 + (Y - center_y_scaled) ** 2)
    mask = torch.clamp(1.0 - torch.clamp(dist - radius_scaled, 0, 1), 0, 1)

    if scale_factor > 1:
        mask = F.avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=scale_factor,
            stride=scale_factor,
        ).squeeze()

    for c in range(C):
        img[:, :, c] = img[:, :, c] + mask * (color[c] - img[:, :, c])

    return img


def gaussian_blur(img, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur using PyTorch convolution."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    H, W, C = img.shape
    x = img.permute(2, 0, 1).unsqueeze(0)

    coords = (
        torch.arange(kernel_size, dtype=torch.float32, device=img.device)
        - kernel_size // 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(C, 1, kernel_size, kernel_size).contiguous()

    padding = kernel_size // 2
    blurred = F.conv2d(x, kernel, padding=padding, groups=C)

    return blurred.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# Physics environment
# ---------------------------------------------------------------------------


class PhysicsControlEnv:
    """Base class for physics environments with discrete action spaces."""

    state_dim: int = 2
    action_dim: int = 3

    def __init__(self, action_map=None):
        if action_map is None:
            action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        self.action_map = action_map
        self.action_dim = len(action_map)

    def step(self, state, action, dt=0.1, variable_params=None):
        raise NotImplementedError

    def get_energy(self, state, variable_params=None):
        raise NotImplementedError

    def render_state(
        self, state, img_size=64, color=True, render_quality="medium",
        ball_color=None, bg_color=None, ball_radius=None,
    ):
        raise NotImplementedError


class ForcedOscillator(PhysicsControlEnv):
    """Forced damped harmonic oscillator.
    State: [x, v] (position, velocity)
    Dynamics: m*a = F_action - c*v - k*x
    """

    state_dim = 2

    def __init__(self, m=1.0, k=1.0, c=0.1, action_map=None):
        super().__init__(action_map=action_map)
        self.m, self.k, self.c = m, k, c

    def step(self, state, action_idx, dt=0.1, variable_params=None):
        if isinstance(action_idx, torch.Tensor):
            action_idx = int(action_idx.item())

        f_val = self.action_map[action_idx]
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            c = variable_params.get("c", self.c)
            m = variable_params.get("m", self.m)
        else:
            k, c, m = self.k, self.c, self.m

        def dynamics(t, s):
            x, v = s[..., 0], s[..., 1]
            dxdt = v
            dvdt = (f_val - (k * x) - (c * v)) / m
            return torch.stack([dxdt, dvdt], dim=-1)

        next_state = odeint(dynamics, state, torch.tensor([0.0, dt]), method="dopri5")[
            -1
        ]
        return next_state

    def get_energy(self, state, variable_params=None):
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            m = variable_params.get("m", self.m)
        else:
            k, m = self.k, self.m
        x, v = state[..., 0], state[..., 1]
        return 0.5 * m * v**2 + 0.5 * k * x**2

    def render_state(
        self, state, img_size=64, color=True, render_quality="medium",
        ball_color=None, bg_color=None, ball_radius=None,
    ):
        world_size = 2.0
        space_res = 2.0 * world_size / img_size
        radius = (
            (ball_radius / space_res) if ball_radius is not None else (self.m / space_res)
        )

        x_pos = state[0].item() if isinstance(state, torch.Tensor) else state[0]

        img = torch.zeros(img_size, img_size, 3)
        bc = torch.tensor(
            ball_color if ball_color is not None else DEFAULT_BALL_COLORS[0]
        )

        px, py = world_to_pixels(0.0, x_pos, img_size, world_size)
        img = render_circle_aa(img, px, py, radius, bc, render_quality)
        img = gaussian_blur(img, kernel_size=5, sigma=1.0)

        bg = torch.tensor(bg_color if bg_color is not None else DEFAULT_BG_COLOR)
        img = img + bg
        img = torch.clamp(img, 0.0, 1.0)

        if not color:
            img = torch.max(img, dim=-1, keepdim=True)[0]

        return img


# ---------------------------------------------------------------------------
# Rollout functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_visual_trajectory(env, init_state, actions, dt, render_opts):
    """Roll out an environment and render each state to an image.

    Args:
        env: PhysicsControlEnv with render_state().
        init_state: (state_dim,) tensor.
        actions: (T,) tensor of discrete action indices.
        dt: timestep for env.step().
        render_opts: dict passed to env.render_state().

    Returns:
        images: (T+1, C, H, W) float tensor in [0, 1].
        states: (T+1, state_dim) tensor.
    """
    states = [init_state]
    state = init_state.clone()
    for t in range(len(actions)):
        state = env.step(state, int(actions[t].item()), dt)
        states.append(state)

    images = []
    for s in states:
        img = env.render_state(s, **render_opts)  # (H, W, C) in [0, 1]
        images.append(img.permute(2, 0, 1))  # (C, H, W)

    return torch.stack(images).float(), torch.stack(states).float()


@torch.no_grad()
def visual_open_loop_rollout(model, images, actions):
    """Open-loop rollout for visual world models with flat latents.

    Encodes all frames, then autoregressively predicts remaining latents.
    Each step the predictor sees context_length latents, produces one
    next-latent, and the window shifts.

    Args:
        model: VisualWorldModel (must conform to model interface contract).
        images: (B, T+1, C, H, W) ground-truth image sequence.
        actions: (B, T) discrete action indices.

    Returns:
        dict with:
            pred_latents: (B, horizon, D_state)
            true_latents: (B, N_latents, D_state)
            pred_images: (B, horizon, C, H, W)
    """
    B, N, C, H, W = images.shape
    ctx_len = model.context_length
    K = model.encoder_frames

    mu_all, _ = model.encode_sequence(images)
    N_latents = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_latents, D_enc)
    true_latents = model.to_state(mu_flat)
    D_state = true_latents.shape[-1]
    true_latents = true_latents.reshape(B, N_latents, D_state)
    horizon = N_latents - ctx_len

    transition_actions = actions[:, K - 1:]

    context = true_latents[:, :ctx_len].clone()

    pred_latents = []
    for t in range(horizon):
        act = transition_actions[:, t : t + ctx_len].long()
        pred = model.predictor(context, act)
        z_next = pred[:, -1]
        pred_latents.append(z_next)
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)

    pred_latents = torch.stack(pred_latents, dim=1)

    pred_images = model.decode(pred_latents.reshape(B * horizon, D_state)).reshape(
        B, horizon, C, H, W
    )

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images": pred_images,
    }


# ---------------------------------------------------------------------------
# Visual metrics
# ---------------------------------------------------------------------------


def mae(pred, target):
    """Mean absolute error."""
    return (pred - target).abs().mean()


def psnr(pred, target, max_val=1.0):
    """Peak signal-to-noise ratio (higher is better)."""
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return (10 * torch.log10(max_val**2 / (mse + 1e-8))).mean()


def _gaussian_kernel(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, window_size=11, sigma=1.5):
    """Structural similarity index (higher is better)."""
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, C, pred.device)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_pp
    sigma_tt = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_tt
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pt

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_pp + mu_tt + c1) * (sigma_pp + sigma_tt + c2)
    )
    return ssim_map.flatten(1).mean()


def compute_visual_metrics(pred_images, true_images):
    """Compute visual reconstruction metrics (no LPIPS).

    Args:
        pred_images: (B, T, C, H, W) in [0, 1].
        true_images: (B, T, C, H, W) in [0, 1].

    Returns:
        dict of scalar metric values.
    """
    B, T, C, H, W = pred_images.shape
    step_mae, step_psnr, step_ssim = [], [], []

    for t in range(T):
        p, g = pred_images[:, t], true_images[:, t]
        step_mae.append(mae(p, g).item())
        step_psnr.append(psnr(p, g).item())
        step_ssim.append(ssim(p, g).item())

    return {
        "mae": np.mean(step_mae),
        "psnr": np.mean(step_psnr),
        "ssim": np.mean(step_ssim),
    }


# ---------------------------------------------------------------------------
# Model interface contract
# ---------------------------------------------------------------------------
#
# The model passed to evaluation functions MUST expose:
#
# Attributes:
#   context_length: int   — number of context frames for prediction
#   encoder_frames: int   — number of frames channel-concatenated for encoding
#   beta: float           — KL weight
#   latent_pred_weight: float — latent prediction loss weight
#
# Methods:
#   encode_sequence(images) -> (mu, logvar)
#       images: (B, T, C, H, W)
#       returns: mu (B, N, D), logvar (B, N, D) where N = T - encoder_frames + 1
#
#   to_state(z) -> s
#       z: (B*N, D_enc)
#       returns: (B*N, D_state)
#
#   reparameterize(mu, logvar) -> s
#       mu, logvar: (B*N, D_enc)
#       returns: (B*N, D_state)
#
#   decode(z) -> images
#       z: (B*N, D_state)
#       returns: (B*N, C, H, W)
#
#   kl_loss(mu, logvar) -> scalar
#
#   predictor(context, actions) -> next_states
#       context: (B, T, D_state)
#       actions: (B, T) long
#       returns: (B, T, D_state)
#


def validate_model_interface(model):
    """Validate that model conforms to the interface contract."""
    assert hasattr(model, "context_length"), "Model must have context_length attribute"
    assert hasattr(model, "encoder_frames"), "Model must have encoder_frames attribute"
    assert hasattr(model, "beta"), "Model must have beta attribute"
    assert hasattr(
        model, "latent_pred_weight"
    ), "Model must have latent_pred_weight attribute"
    assert callable(
        getattr(model, "encode_sequence", None)
    ), "Model must have encode_sequence method"
    assert callable(
        getattr(model, "to_state", None)
    ), "Model must have to_state method"
    assert callable(
        getattr(model, "reparameterize", None)
    ), "Model must have reparameterize method"
    assert callable(getattr(model, "decode", None)), "Model must have decode method"
    assert callable(
        getattr(model, "kl_loss", None)
    ), "Model must have kl_loss method"
    assert callable(
        getattr(model, "predictor", None)
    ), "Model must have predictor (callable)"


# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_val_loss(model, val_loader, device):
    """Run one pass over validation set, return loss dict.

    Uses HGN-style evaluation: encode all frames, predict next states,
    decode predicted states, compute reconstruction + KL + latent pred losses.
    """
    model.eval()

    total_recon = 0.0
    total_kl = 0.0
    total_latent_pred = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = batch_to_device(batch, device)
        images = batch["images"]
        actions = batch["actions"]
        B, _, C, H, W = images.shape
        K = model.encoder_frames
        ctx_len = model.context_length
        pred_len = getattr(model, "pred_length", 1)

        mu_all, logvar_all = model.encode_sequence(images)
        N_lat = mu_all.shape[1]
        D_enc = mu_all.shape[2]

        mu_flat = mu_all.reshape(B * N_lat, D_enc)
        logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
        all_states = model.to_state(mu_flat)
        D_state = all_states.shape[-1]
        all_states = all_states.reshape(B, N_lat, D_state)

        kl_loss = model.kl_loss(mu_flat, logvar_flat).item()
        total_kl += kl_loss

        transition_actions = actions[:, K - 1:]
        window_size = ctx_len + pred_len
        step_size = pred_len
        num_windows = max(1, 1 + (N_lat - window_size) // step_size)

        recon_loss = 0.0
        latent_pred_loss = 0.0
        for w in range(num_windows):
            start = w * step_size
            end = min(start + window_size, N_lat)
            w_states = all_states[:, start:end]
            n_pred = w_states.shape[1] - 1

            pred_input = w_states[:, :-1]
            w_actions = transition_actions[:, start : start + n_pred].long()
            pred_z = model.predictor(pred_input, w_actions)

            pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
            gt_start = K - 1 + start + 1
            gt_frames = images[:, gt_start : gt_start + n_pred].reshape(
                B * n_pred, C, H, W
            )
            recon_loss += ((pred_decoded - gt_frames) ** 2).mean().item() / num_windows

            target_states = w_states[:, 1:]
            latent_pred_loss += (
                ((pred_z - target_states) ** 2).mean().item() / num_windows
            )

        total_recon += recon_loss
        total_latent_pred += latent_pred_loss
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_recon_loss": total_recon / n,
        "val_kl_loss": total_kl / n,
        "val_latent_pred_loss": total_latent_pred / n,
        "val_total_loss": (total_recon + model.beta * total_kl + model.latent_pred_weight * total_latent_pred) / n,
    }


@torch.no_grad()
def evaluate_dt_generalization(model, device):
    """Compute val_dt_score: average latent MSE across dt generalization tests.

    For each dt in EVAL_DT_VALUES:
        - Generate EVAL_N_SEQS fresh trajectories from the oscillator environment
        - Encode all frames, run open-loop autoregressive rollout
        - Compute latent MSE between predicted and encoded ground-truth latents

    Returns:
        val_dt_score: float (lower is better)
        dt_breakdown: dict mapping dt -> latent_mse
    """
    model.eval()

    # Fixed seed for reproducibility
    rng_state = torch.random.get_rng_state()
    np_rng_state = np.random.get_state()
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    env = ForcedOscillator(m=1.0, k=1.0, c=0.1)
    render_opts = {"img_size": IMG_SIZE, "color": True, "render_quality": "medium"}
    init_range = np.array([[-2.0, 2.0], [-2.0, 2.0]])  # [x, v] ranges

    dt_breakdown = {}

    for dt in EVAL_DT_VALUES:
        all_images = []
        all_actions = []
        for _ in range(EVAL_N_SEQS):
            init_state = torch.tensor(
                [np.random.uniform(r[0], r[1]) for r in init_range]
            ).float()
            actions = torch.randint(0, ACTION_DIM, (EVAL_SEQ_LEN,))
            imgs, _ = generate_visual_trajectory(env, init_state, actions, dt, render_opts)
            all_images.append(imgs)
            all_actions.append(actions)

        images_batch = torch.stack(all_images).to(device)
        actions_batch = torch.stack(all_actions).to(device)

        rollout = visual_open_loop_rollout(model, images_batch, actions_batch)
        pred_latents = rollout["pred_latents"]
        true_latents = rollout["true_latents"]

        ctx_len = model.context_length
        gt_latents = true_latents[:, ctx_len:]
        latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
        dt_breakdown[dt] = latent_mse

    # Restore RNG state
    torch.random.set_rng_state(rng_state)
    np.random.set_state(np_rng_state)

    val_dt_score = np.mean(list(dt_breakdown.values()))
    return val_dt_score, dt_breakdown


# ---------------------------------------------------------------------------
# Main: verify data and print stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Data root: {DATA_ROOT}")
    print(f"Dataset: {DATASET_NAME}/{DATASET_VERSION}")
    print(f"Dataset path: {DATASET_PATH}")
    print()

    # Check files exist
    train_path = os.path.join(DATASET_PATH, "train.npz")
    val_path = os.path.join(DATASET_PATH, "val.npz")

    for path, name in [(train_path, "train"), (val_path, "val")]:
        if os.path.exists(path):
            data = np.load(path, mmap_mode="r")
            n_seqs = data["states"].shape[0]
            seq_len = data["states"].shape[1] - 1
            has_images = "images" in data
            img_shape = data["images"].shape[2:] if has_images else None
            print(f"  {name}: {n_seqs} sequences, seq_len={seq_len}, images={has_images} {img_shape or ''}")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    print()
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Eval dt values: {EVAL_DT_VALUES}")
    print(f"Eval sequences per dt: {EVAL_N_SEQS}")
    print()
    print("Ready to train.")
