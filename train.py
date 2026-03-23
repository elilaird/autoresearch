"""
Autoresearch training script for physics world models.
Single-GPU, single-file. This is the ONLY file you modify.

Usage: python train.py

Architecture: Beta-VAE visual world model with Hamiltonian predictor
and Transformer temporal backbone. Learns physics dynamics in latent
space from 64x64 rendered oscillator images.

The model is evaluated on dt generalization: how well learned dynamics
transfer across different sampling rates (dt=0.1, 0.2, 0.5).
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from prepare import (
    TIME_BUDGET,
    ACTION_DIM,
    make_dataloaders,
    batch_to_device,
    evaluate_dt_generalization,
    evaluate_val_loss,
    validate_model_interface,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Architecture
LATENT_CHANNELS = 64        # total latent dim (split into position + momentum halves)
HIDDEN_CHANNELS = 512       # hidden dim for encoder/decoder MLPs
BETA = 0.003                # KL divergence weight
FREE_BITS = 0.0             # per-element KL floor (prevents posterior collapse)
CONTEXT_LENGTH = 3          # number of context frames for predictor
PRED_LENGTH = 1             # number of frames to predict per window
LATENT_PRED_WEIGHT = 1.0    # weight of latent prediction loss
ENCODER_FRAMES = 2          # frames channel-concatenated for velocity estimation

# Predictor
PREDICTOR_TYPE = "hamiltonian"  # "mlp", "lstm", "transformer", "ode", "newtonian", "hamiltonian"
PREDICTOR_HIDDEN = 256      # predictor hidden dimension
ACTION_EMBEDDING_DIM = 8    # action embedding dimension
INTEGRATION_DT = 0.2        # ODE integration timestep (should match training data dt)
INTEGRATION_METHOD = "rk4"  # ODE solver: "euler", "rk4", "dopri5"
DAMPING_INIT = -1.0         # initial log-damping for Newtonian/Hamiltonian
BACKBONE = "transformer"    # temporal backbone: None, "lstm", "transformer"
BACKBONE_LAYERS = 2         # number of backbone layers
BACKBONE_NHEAD = 4          # transformer attention heads

# Optimization
BATCH_SIZE = 32
LR = 1.5e-4
SEED = 42

# Logging
WANDB_PROJECT = "autoresearch"
WANDB_ENABLED = HAS_WANDB and os.environ.get("WANDB_DISABLED", "").lower() != "true"

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    """Residual block: two 3x3 convs with LeakyReLU and skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Vision encoder / decoder
# ---------------------------------------------------------------------------


class VisionEncoder(nn.Module):

    def __init__(self, channels=3, latent_channels=32, encoder_frames=1, hidden_channels=512):
        super().__init__()
        in_channels = channels * encoder_frames
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 64x64
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 64->32
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 32x32
            nn.Conv2d(64, 64, 4, 2, 1),  # 32->16
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 16x16
            nn.Conv2d(64, 64, 4, 2, 1),  # 16->8
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 8x8
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, latent_channels * 2),
        )

    def forward(self, x):
        return self.mlp(self.cnn(x).flatten(1)).chunk(2, dim=-1)


class VisionDecoder(nn.Module):
    """Decodes flat (B, D_q) latents to (B, C, 64, 64) images."""

    def __init__(self, channels=3, latent_channels=16, hidden_channels=512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 64 * 8 * 8),
        )
        self.cnn = nn.Sequential(
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8->16
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16->32
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32->64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.project(z).reshape(z.shape[0], 64, 8, 8)
        return self.cnn(h)


def kl_divergence_free_bits(mu, logvar, free_bits=0.5):
    """KL divergence with free bits (per-element clamping)."""
    kl_per_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    kl_clamped = torch.clamp(kl_per_elem, min=free_bits)
    return kl_clamped.flatten(1).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Temporal backbone
# ---------------------------------------------------------------------------


class TemporalBackbone(nn.Module):
    """Sequence model backbone (LSTM or Transformer) for temporal context."""

    def __init__(self, input_dim, hidden_dim, backbone_type="lstm", num_layers=2, nhead=4):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "lstm":
            self.net = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif backbone_type == "transformer":
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(self, x):
        if self.backbone_type == "lstm":
            out, _ = self.net(x)
            return out
        else:
            x = self.proj_in(x)
            T = x.shape[1]
            # Float additive causal mask (required for autograd compatibility)
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            return self.net(x, mask=mask)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


PREDICTOR_REGISTRY = {}


def register_predictor(name):
    def decorator(cls):
        PREDICTOR_REGISTRY[name] = cls
        return cls
    return decorator


@register_predictor("mlp")
class MLPPredictor(nn.Module):
    """Per-frame residual MLP: z_{t+1} = z_t + f(z_t, a_t)."""

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim, **kw):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context, actions):
        emb = self.act_emb(actions)
        x = torch.cat([context, emb], dim=-1)
        return context + self.net(x)


@register_predictor("lstm")
class LSTMPredictor(nn.Module):
    """LSTM over context sequence with residual output."""

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim, num_layers=2, **kw):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context, actions):
        emb = self.act_emb(actions)
        x = torch.cat([context, emb], dim=-1)
        out, _ = self.lstm(x)
        return context + self.output(out)


@register_predictor("transformer")
class TransformerPredictor(nn.Module):
    """Causal Transformer over context sequence with residual output."""

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim, num_layers=2, nhead=4, **kw):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.proj_in = nn.Linear(latent_dim + action_embedding_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context, actions):
        emb = self.act_emb(actions)
        x = self.proj_in(torch.cat([context, emb], dim=-1))
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        x = self.transformer(x, mask=mask)
        return context + self.proj_out(x)


@register_predictor("ode")
class ODEPredictor(nn.Module):
    """First-order neural ODE: dz/dt = f(z, a)."""

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim,
                 dt=0.1, integration_method="rk4", backbone=None,
                 backbone_layers=2, backbone_nhead=4, **kw):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                latent_dim + action_embedding_dim, hidden_dim,
                backbone, backbone_layers, backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + conditioning_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        return self.net(torch.cat([z, self._conditioning_cache], dim=-1))

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)
            features = self.backbone(inp)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)


@register_predictor("newtonian")
class NewtonianPredictor(nn.Module):
    """Newtonian dynamics: dq/dt = p, dp/dt = f(q, p, a) - gamma*p."""

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim,
                 dt=0.1, integration_method="rk4", damping_init=-1.0,
                 backbone=None, backbone_layers=2, backbone_nhead=4, **kw):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                latent_dim + action_embedding_dim, hidden_dim,
                backbone, backbone_layers, backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.accel_net = nn.Sequential(
            nn.Linear(latent_dim + conditioning_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.half_dim),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        q, p = z[..., :self.half_dim], z[..., self.half_dim:]
        damping = F.softplus(self.log_damping)
        accel = self.accel_net(torch.cat([z, self._conditioning_cache], dim=-1))
        dq = p
        dp = accel - damping * p
        return torch.cat([dq, dp], dim=-1)

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)
            features = self.backbone(inp)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)


@register_predictor("hamiltonian")
class HamiltonianPredictor(nn.Module):
    """Port-Hamiltonian predictor: learns H(q, p), derives dynamics via autograd.

    Symplectic structure: dq/dt = dH/dp, dp/dt = -dH/dq.
    Includes dissipation (learned damping) and input port G(a) for actions.
    Full dynamics: dq/dt = dH/dp, dp/dt = -dH/dq - gamma*dH/dp + G(a).
    """

    def __init__(self, latent_dim, action_dim, action_embedding_dim, hidden_dim,
                 dt=0.1, integration_method="rk4", damping_init=-1.0,
                 backbone=None, backbone_layers=2, backbone_nhead=4, **kw):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                latent_dim + action_embedding_dim, hidden_dim,
                backbone, backbone_layers, backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.G_net = nn.Linear(conditioning_dim, self.half_dim)
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)

        with torch.enable_grad():
            H = self.H_net(z).sum()
            dH = torch.autograd.grad(H, z, create_graph=True)[0]

        dH_dq = dH[..., :self.half_dim]
        dH_dp = dH[..., self.half_dim:]

        damping = F.softplus(self.log_damping)
        G_u = self.G_net(self._conditioning_cache)

        dq = dH_dp
        dp = -dH_dq - damping * dH_dp + G_u
        return torch.cat([dq, dp], dim=-1)

    def energy(self, z):
        """Compute Hamiltonian energy for monitoring."""
        return self.H_net(z)

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)
            features = self.backbone(inp)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Visual world model
# ---------------------------------------------------------------------------


class VisualWorldModel(nn.Module):
    """Beta-VAE encoder/decoder + swappable flat latent-space predictor.

    Latent space: z in (B, D) where D = latent_channels.
    Structured as z = [z_q, z_p] split on last dim.
    z_q (position, first half) drives decoding;
    z_p (momentum, second half) carries dynamics information.
    """

    def __init__(self, predictor, latent_channels, hidden_channels, beta,
                 free_bits, context_length, pred_length, latent_pred_weight,
                 encoder_frames, channels=3):
        super().__init__()
        assert latent_channels % 2 == 0
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.beta = beta
        self.free_bits = free_bits
        self.context_length = context_length
        self.pred_length = pred_length
        self.latent_pred_weight = latent_pred_weight
        self.encoder_frames = encoder_frames
        self.channels = channels

        self.encoder = VisionEncoder(
            channels=channels,
            latent_channels=latent_channels,
            encoder_frames=encoder_frames,
            hidden_channels=hidden_channels,
        )
        self.decoder = VisionDecoder(
            channels=channels,
            latent_channels=latent_channels // 2,
            hidden_channels=hidden_channels,
        )
        self.predictor = predictor

        self.state_transform = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, latent_channels),
        )

    def encode(self, images):
        mu, logvar = self.encoder(images)
        return mu, logvar

    def encode_sequence(self, images):
        """Encode frame sequence using overlapping channel-concatenated windows.

        Args:
            images: (B, T, C, H, W)
        Returns:
            mu, logvar: each (B, T - encoder_frames + 1, latent_channels)
        """
        B, T, C, H, W = images.shape
        K = self.encoder_frames
        n_out = T - K + 1
        windows = torch.cat(
            [images[:, t:t + K].reshape(B, K * C, H, W).unsqueeze(1) for t in range(n_out)],
            dim=1,
        )
        catted = windows.reshape(B * n_out, K * C, H, W)
        mu, logvar = self.encode(catted)
        D = mu.shape[-1]
        return mu.reshape(B, n_out, D), logvar.reshape(B, n_out, D)

    def reparameterize(self, mu, logvar):
        """Sample z and map to phase-space state."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.state_transform(z)

    def to_state(self, z):
        return self.state_transform(z)

    def decode(self, z):
        return self.decoder(z[..., :self.latent_channels // 2])

    def kl_loss(self, mu, logvar):
        return kl_divergence_free_bits(mu, logvar, self.free_bits)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_predictor():
    cls = PREDICTOR_REGISTRY[PREDICTOR_TYPE]
    kwargs = dict(
        latent_dim=LATENT_CHANNELS,
        action_dim=ACTION_DIM,
        action_embedding_dim=ACTION_EMBEDDING_DIM,
        hidden_dim=PREDICTOR_HIDDEN,
    )
    # ODE-based predictors need extra args
    if PREDICTOR_TYPE in ("ode", "newtonian", "hamiltonian"):
        kwargs.update(
            dt=INTEGRATION_DT,
            integration_method=INTEGRATION_METHOD,
            backbone=BACKBONE,
            backbone_layers=BACKBONE_LAYERS,
            backbone_nhead=BACKBONE_NHEAD,
        )
    if PREDICTOR_TYPE in ("newtonian", "hamiltonian"):
        kwargs["damping_init"] = DAMPING_INIT
    return cls(**kwargs)


def build_model():
    predictor = build_predictor()
    return VisualWorldModel(
        predictor=predictor,
        latent_channels=LATENT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        beta=BETA,
        free_bits=FREE_BITS,
        context_length=CONTEXT_LENGTH,
        pred_length=PRED_LENGTH,
        latent_pred_weight=LATENT_PRED_WEIGHT,
        encoder_frames=ENCODER_FRAMES,
    )


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------


def _has_energy(predictor):
    return hasattr(predictor, "energy") and callable(predictor.energy)


def hgn_train_step(model, batch, optimizer):
    """HGN training step with sliding-window predictor."""
    images = batch["images"]
    actions = batch["actions"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    mu_all, logvar_all = model.encode_sequence(images)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
    all_states = model.reparameterize(mu_flat, logvar_flat)
    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    transition_actions = actions[:, K - 1:]

    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    recon_loss = torch.tensor(0.0, device=images.device)
    latent_pred_loss = torch.tensor(0.0, device=images.device)
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss = recon_loss + ((pred_decoded - gt_frames) ** 2).mean() / num_windows

        target_states = w_states[:, 1:].detach()
        latent_pred_loss = latent_pred_loss + ((pred_z - target_states) ** 2).mean() / num_windows

    kl_loss = model.kl_loss(mu_flat, logvar_flat)

    loss = recon_loss + model.beta * kl_loss + model.latent_pred_weight * latent_pred_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "latent_pred_loss": latent_pred_loss.item(),
        "total_loss": loss.item(),
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model().to(device)
validate_model_interface(model)

num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: VisualWorldModel / {PREDICTOR_TYPE} (backbone={BACKBONE})")
print(f"Parameters: {num_params:,} total, {trainable_params:,} trainable")

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR
)

train_loader, val_loader = make_dataloaders(BATCH_SIZE)
print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------

if WANDB_ENABLED:
    # Get git info for run name
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
        git_branch = "unknown"

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{PREDICTOR_TYPE}/{BACKBONE or 'none'}/{git_hash}",
        tags=[PREDICTOR_TYPE, BACKBONE or "no-backbone", git_branch],
        config={
            "latent_channels": LATENT_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "beta": BETA,
            "free_bits": FREE_BITS,
            "context_length": CONTEXT_LENGTH,
            "pred_length": PRED_LENGTH,
            "latent_pred_weight": LATENT_PRED_WEIGHT,
            "encoder_frames": ENCODER_FRAMES,
            "predictor_type": PREDICTOR_TYPE,
            "predictor_hidden": PREDICTOR_HIDDEN,
            "action_embedding_dim": ACTION_EMBEDDING_DIM,
            "integration_dt": INTEGRATION_DT,
            "integration_method": INTEGRATION_METHOD,
            "damping_init": DAMPING_INIT,
            "backbone": BACKBONE,
            "backbone_layers": BACKBONE_LAYERS,
            "backbone_nhead": BACKBONE_NHEAD,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "num_params_M": num_params / 1e6,
            "git_hash": git_hash,
            "git_branch": git_branch,
        },
    )
    print(f"Wandb run: {wandb.run.url}")

# ---------------------------------------------------------------------------
# Training loop (time-budgeted)
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
step = 0
epoch = 0
smooth_loss = 0
warmup_steps = 3  # exclude first steps from time counting (JIT warmup)

while True:
    model.train()
    epoch += 1

    for batch in train_loader:
        torch.cuda.synchronize()
        t0 = time.time()

        batch = batch_to_device(batch, device)
        losses = hgn_train_step(model, batch, optimizer)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step >= warmup_steps:
            total_training_time += dt

        train_loss = losses["total_loss"]

        # Fast fail
        if math.isnan(train_loss) or train_loss > 1000:
            print("FAIL: loss exploded or NaN")
            exit(1)

        # EMA smoothing
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)

        if WANDB_ENABLED:
            wandb.log({
                "train/total_loss": train_loss,
                "train/recon_loss": losses["recon_loss"],
                "train/kl_loss": losses["kl_loss"],
                "train/latent_pred_loss": losses["latent_pred_loss"],
                "train/smooth_loss": debiased_loss,
                "progress": progress,
            }, step=step)

        if step % 50 == 0:
            print(
                f"\rstep {step:05d} ({100*progress:.1f}%) | "
                f"loss: {debiased_loss:.6f} | "
                f"recon: {losses['recon_loss']:.4f} | "
                f"kl: {losses['kl_loss']:.4f} | "
                f"pred: {losses['latent_pred_loss']:.4f} | "
                f"epoch: {epoch} | "
                f"remaining: {remaining:.0f}s    ",
                end="", flush=True,
            )

        step += 1

        if step > warmup_steps and total_training_time >= TIME_BUDGET:
            break

    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()  # newline after progress
print(f"Training complete: {step} steps, {epoch} epochs, {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("\nEvaluating on validation set...")
val_metrics = evaluate_val_loss(model, val_loader, device)
for k, v in val_metrics.items():
    print(f"  {k}: {v:.6f}")

print("\nEvaluating dt generalization...")
val_dt_score, dt_breakdown = evaluate_dt_generalization(model, device)
for dt_val, mse in sorted(dt_breakdown.items()):
    print(f"  dt={dt_val}: latent_mse={mse:.6f}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_dt_score:      {val_dt_score:.6f}")
print(f"val_recon_loss:    {val_metrics['val_recon_loss']:.6f}")
print(f"val_kl_loss:       {val_metrics['val_kl_loss']:.6f}")
print(f"val_latent_pred:   {val_metrics['val_latent_pred_loss']:.6f}")
for dt_val in sorted(dt_breakdown.keys()):
    print(f"dt_{dt_val}_mse:       {dt_breakdown[dt_val]:.6f}")
print(f"training_seconds:  {total_training_time:.1f}")
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"num_epochs:        {epoch}")
print(f"num_steps:         {step}")
print(f"num_params_M:      {num_params / 1e6:.1f}")
print(f"predictor_type:    {PREDICTOR_TYPE}")
print(f"backbone:          {BACKBONE}")

if WANDB_ENABLED:
    summary = {
        "val/dt_score": val_dt_score,
        "val/recon_loss": val_metrics["val_recon_loss"],
        "val/kl_loss": val_metrics["val_kl_loss"],
        "val/latent_pred_loss": val_metrics["val_latent_pred_loss"],
        "peak_vram_mb": peak_vram_mb,
        "num_epochs": epoch,
        "num_steps": step,
    }
    for dt_val, mse in dt_breakdown.items():
        summary[f"val/dt_{dt_val}_mse"] = mse
    wandb.log(summary, step=step)
    wandb.finish()
