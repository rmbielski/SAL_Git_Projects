# cheatgrass_model.py
# ---------------------------------------------------------------------
# Phenology-aware sequence segmentation (PyTorch) with robust cropping
# Callable API: train(), evaluate(), build_model(), make_dataloaders()
# ---------------------------------------------------------------------

import os
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from datetime import datetime
import uuid


# -----------------------------
# Config & Utilities
# -----------------------------
@dataclass
class ModelConfig:
    data_dir: Path
    output_dir: Path
    batch_size: int = 1
    learning_rate: float = 1e-4
    epochs: int = 50
    hidden_dim: int = 512  # more capacity for temporal modeling
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    attn_heads: int = 16
    input_bands: int = 6
    train_val_split_size: float = 0.2
    training_window_size: int = 96
    enable_augmentation: bool = True
    train_crops_per_sample: int = 100
    validation_crops_per_sample: int = 100
    train_negative_crop_prob: float = 0.2   # small amount of background crops to calibrate
    validation_negative_crop_prob: float = 0.1
    eval_negative_crop_prob: float = 0.0
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reproducibility
    seed: int = 42
    # dataloader
    num_workers: int = 0
    pin_memory: bool = False
    # memory-saving options
    use_checkpointing: bool = True

    # NEW: debugging / verbose options for augmentation/training pipeline
    verbose: bool = False
    debug_every_n_batches: int = 50  # print debug crop info every N batches

    def prepare(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self):
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        d["device"] = str(self.device)
        return d



def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.device):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    # fall back to str for anything else unexpected
    return str(o)

# NEW: Ensure nested objects are JSON primitives
def make_json_safe(obj):
    """Recursively convert non-JSON primitives to JSON-friendly types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return [make_json_safe(x) for x in obj.flat]
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    # Pandas conversions
    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Series, _pd.Index)):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass
    # Fallback to string
    return str(obj)



# -----------------------------
# Loss
# -----------------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits; targets: {0,1} same shape
        inputs = torch.sigmoid(inputs)
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - tversky) ** self.gamma

# NEW: Weighted BCE with logits
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logits, targets):
        # logits, targets flattened equally
        probs = torch.sigmoid(logits)
        loss_pos = -targets * torch.log(probs + 1e-6) * self.pos_weight
        loss_neg = -(1 - targets) * torch.log(1 - probs + 1e-6) * self.neg_weight
        return (loss_pos + loss_neg).mean()

# NEW: Combo loss (Tversky + Weighted BCE)
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, bce_pos=1.0, bce_neg=3.0, mix=0.5):
        super().__init__()
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = WeightedBCEWithLogitsLoss(pos_weight=bce_pos, neg_weight=bce_neg)
        self.mix = mix  # 0..1; 1=tversky only

    def forward(self, logits, targets, valid_mask: Optional[torch.Tensor] = None):
        ft = logits.flatten()
        tt = targets.flatten()
        if valid_mask is not None:
            vm = valid_mask.flatten()
            keep = vm > 0.5
            if not keep.any():
                # Zero loss but keep graph attached to logits so backward is safe
                return (logits * 0.0).sum()
            ft = ft[keep]
            tt = tt[keep]
        return self.mix * self.tversky(ft, tt) + (1 - self.mix) * self.bce(ft, tt)

# Tiny sparsity regularizer weight
SPARSITY_LAMBDA = 1e-4  # softened: allow higher probabilities

# Channel-wise normalization stats (computed on cheatgrass_data_validmask_copy/train after masking & clamping)
BAND_MEAN = torch.tensor([224.4384, 311.10776, 343.74658, 744.1865, 570.997, 854.45917], dtype=torch.float32)
BAND_STD = torch.tensor([536.07794, 611.5151, 664.91455, 1108.6785, 901.4944, 1299.8438], dtype=torch.float32)


# -----------------------------
# Cropping helpers
# -----------------------------
def load_sample_metadata(data_dir: Path, sample_id: str) -> Optional[Dict[str, Any]]:
    p = data_dir / f"{sample_id}_metadata.json"
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def find_valid_crop_position_enhanced(
    mask: np.ndarray,
    crop_size: int,
    force_include_vegetation: bool = True,
    min_mask_pixels: int = 1,
) -> Tuple[int, int]:
    """
    mask: (H, W) binary/float
    """
    h, w = mask.shape
    if h <= crop_size or w <= crop_size:
        return 0, 0

    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or not force_include_vegetation:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return top, left

    max_attempts = 100
    valid_positions = []
    for _ in range(max_attempts):
        i = random.randint(0, len(ys) - 1)
        ay, ax = ys[i], xs[i]

        r = random.random()
        if r < 0.2:
            oy = random.randint(0, crop_size // 4)
            ox = random.randint(0, crop_size // 4)
        elif r < 0.4:
            oy = random.randint(0, crop_size // 4)
            ox = random.randint(3 * crop_size // 4, crop_size - 1)
        elif r < 0.6:
            oy = random.randint(3 * crop_size // 4, crop_size - 1)
            ox = random.randint(0, crop_size // 4)
        elif r < 0.8:
            oy = random.randint(3 * crop_size // 4, crop_size - 1)
            ox = random.randint(3 * crop_size // 4, crop_size - 1)
        else:
            oy = random.randint(0, crop_size - 1)
            ox = random.randint(0, crop_size - 1)

        top = max(0, min(ay - oy, h - crop_size))
        left = max(0, min(ax - ox, w - crop_size))

        if mask[top : top + crop_size, left : left + crop_size].sum() >= min_mask_pixels:
            valid_positions.append((top, left))
            if len(valid_positions) >= 5:
                break

    if valid_positions:
        return random.choice(valid_positions)

    for _ in range(20):
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        if mask[top : top + crop_size, left : left + crop_size].sum() >= min_mask_pixels:
            return top, left

    return (h - crop_size) // 2, (w - crop_size) // 2

# NEW: explicit pure-background crop finder
def find_pure_background_crop(
    mask: np.ndarray,
    crop_size: int,
    max_attempts: int = 100,
) -> Tuple[int, int]:
    """Return a crop with zero positive pixels if possible; else random."""
    h, w = mask.shape
    if h <= crop_size or w <= crop_size:
        return 0, 0
    for _ in range(max_attempts):
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        if mask[top:top+crop_size, left:left+crop_size].sum() == 0:
            return top, left
    # fallback random
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return top, left


def enhanced_random_crop_sample(
    data: torch.Tensor,   # (T, C, H, W)
    mask: torch.Tensor,   # (T, 1, H, W) or (1, 1, H, W)
    crop_size: int,
    is_training: bool = True,
    force_spatial_diversity: bool = True,
    debug: bool = False,
    # NEW: allow overriding force_include_vegetation to enable negative crops
    force_include_vegetation: Optional[bool] = None,
    valid: Optional[torch.Tensor] = None,  # (T,1,H,W) or (1,1,H,W)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Tuple[int, int, bool, int]]]:
    t, c, h, w = data.shape
    if h == crop_size and w == crop_size:
        if debug:
            # no augmentation needed: returns center coords & mask sum
            mask_sum = int(mask.sum().item())
            if valid is not None:
                return data, mask, valid, ((h - crop_size)//2, (w - crop_size)//2, False, mask_sum)
            return data, mask, None, ((h - crop_size)//2, (w - crop_size)//2, False, mask_sum)
        if valid is not None:
            return data, mask, valid, None
        return data, mask, None, None

    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        data = torch.nn.functional.pad(data, (0, pad_w, 0, pad_h))
        mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
        if valid is not None:
            valid = torch.nn.functional.pad(valid, (0, pad_w, 0, pad_h))
        _, _, h, w = data.shape

    if (is_training and force_spatial_diversity) or (h > crop_size and w > crop_size):
        # Use any positive across time, not just the first slice (masks can be time-varying)
        union_mask = (mask > 0).any(dim=0)[0].detach().cpu().numpy()
        # if force_include_vegetation is None -> default True (prior behavior)
        must_include = True if force_include_vegetation is None else bool(force_include_vegetation)
        if must_include:
            top, left = find_valid_crop_position_enhanced(
                union_mask, crop_size, force_include_vegetation=True
            )
        else:
            top, left = find_pure_background_crop(union_mask, crop_size)
        augmented = True
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        augmented = False

    dc = data[:, :, top : top + crop_size, left : left + crop_size]
    mc = mask[:, :, top : top + crop_size, left : left + crop_size]
    vc = None
    if valid is not None:
        vc = valid[:, :, top : top + crop_size, left : left + crop_size]

    if debug:
        mask_sum = int(mc.sum().item())
        return dc, mc, vc, (top, left, bool(augmented), mask_sum)
    return dc, mc, vc, None


# -----------------------------
# Model
# -----------------------------
class PixelTemporalPhenology(nn.Module):
    """
    Per-pixel temporal/band model with minimal spatial bias.
    Flattens spatial dimensions; runs temporal conv + Transformer over time; predicts per-time logits for each pixel.
    """
    def __init__(
        self,
        input_bands: int = 6,
        hidden_dim: int = 256,
        n_heads: int = 8,
        ff_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = bool(use_checkpointing)
        self.embed = nn.Linear(input_bands, hidden_dim)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # (B,H,W,T,C)
        x = x.reshape(b * h * w, t, c)  # (B*H*W, T, C)
        x = self.embed(x)               # (B*H*W, T, hidden)
        x_tc = x.transpose(1, 2)        # (B*H*W, hidden, T)
        x_tc = self.temporal_conv(x_tc)
        x_tc = x_tc.transpose(1, 2)     # (B*H*W, T, hidden)
        if self.use_checkpointing:
            x_tc = x_tc.requires_grad_()
            # Recompute each encoder layer during backward to save activations
            for layer in self.transformer.layers:
                x_tc = checkpoint(layer, x_tc, use_reentrant=False)
            x_enc = x_tc
        else:
            x_enc = self.transformer(x_tc)  # (B*H*W, T, hidden)
        logits = self.head(x_enc)       # (B*H*W, T, 1)
        logits = logits.view(b, h, w, t, 1).permute(0, 3, 4, 1, 2)  # (B,T,1,H,W)
        return logits


# -----------------------------
# Dataset
# -----------------------------
class SpatiallyRobustVeduDataset(Dataset):
    """
    Expects files:
      {id}_data.npy  -> shape (T, H, W, C)
      {id}_mask.npy  -> shape (T, H, W) or (H, W)
    Returns tensors:
      data: (T, C, 96, 96), mask: (T, 1, 96, 96)
    """
    def __init__(
        self,
        data_dir: Path,
        location_ids: List[str],
        training_window_size: int = 96,
        is_training: bool = True,
        enable_augmentation: bool = True,
        crops_per_epoch: int = 1,
        seed: int = 42,
        verbose: bool = False,
        debug_every_n_batches: int = 50,
        # NEW: probability to take a pure-background crop (no forced vegetation)
        negative_crop_prob: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        self.location_ids = [str(g) for g in location_ids]
        self.training_window_size = training_window_size
        self.is_training = is_training
        self.enable_augmentation = enable_augmentation
        self.crops_per_epoch = crops_per_epoch
        self.current_epoch = 0
        self.seed = seed
        self.verbose = bool(verbose)
        self.debug_every_n_batches = int(debug_every_n_batches)
        self.last_debug_info = None  # holds last per-sample debug info
        self.sample_metadata = {gid: load_sample_metadata(self.data_dir, gid) for gid in self.location_ids}
        self.negative_crop_prob = float(negative_crop_prob)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self) -> int:
        return len(self.location_ids) * self.crops_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        random.seed(idx + self.current_epoch * 1151 + self.seed)

        sample_idx = idx // self.crops_per_epoch
        gid = self.location_ids[sample_idx]

        data = np.load(self.data_dir / f"{gid}_data.npy").astype(np.float32)  # (T,H,W,C)
        mask = np.load(self.data_dir / f"{gid}_mask.npy").astype(np.float32)  # (T,H,W) or (H,W)
        # NEW: validity mask
        valid_path = self.data_dir / f"{gid}_valid.npy"
        if valid_path.exists():
            valid = np.load(valid_path).astype(np.float32)  # (T,H,W) or (H,W)
        else:
            valid = np.ones_like(mask, dtype=np.float32)
        # Reorder data
        data = np.transpose(data, (0, 3, 1, 2))
        if mask.ndim == 3:
            mask = np.expand_dims(mask, 1)
        elif mask.ndim == 2:
            mask = mask[None, None, ...]
        if valid.ndim == 3:
            valid = np.expand_dims(valid, 1)
        elif valid.ndim == 2:
            valid = valid[None, None, ...]

        data_t = torch.from_numpy(data)
        mask_t = torch.from_numpy(mask)
        valid_t = torch.from_numpy(valid)

        if self.enable_augmentation or (data_t.shape[2], data_t.shape[3]) != (self.training_window_size, self.training_window_size):
            force_include = True
            if random.random() < self.negative_crop_prob:
                force_include = False
            dc, mc, vc, debug_info = enhanced_random_crop_sample(
                data_t, mask_t, self.training_window_size,
                is_training=self.is_training,
                force_spatial_diversity=self.enable_augmentation,
                debug=self.verbose,
                force_include_vegetation=force_include,
                valid=valid_t,
             )
            data_t, mask_t, valid_t = dc, mc, vc
        else:
            debug_info = None

        gid = self.location_ids[sample_idx]
        if self.verbose:
            top, left, aug_flag, mask_sum = debug_info if debug_info is not None else ((data_t.shape[2]-self.training_window_size)//2, (data_t.shape[3]-self.training_window_size)//2, False, int(mask_t.sum().item()))
            self.last_debug_info = {
                "gid": gid,
                "top": int(top),
                "left": int(left),
                "augmentation": bool(aug_flag),
                "mask_sum_in_crop": int(mask_sum),
                "crop_size": int(self.training_window_size),
            }

        # Guard: detect and sanitize non-finite values, report gid
        if not torch.isfinite(data_t).all() or not torch.isfinite(mask_t).all() or not torch.isfinite(valid_t).all():
            print(f"[data warning] non-finite values in sample {gid}; sanitizing with nan_to_num")
            data_t = torch.nan_to_num(data_t, nan=0.0, posinf=0.0, neginf=0.0)
            mask_t = torch.nan_to_num(mask_t, nan=0.0, posinf=0.0, neginf=0.0)
            valid_t = torch.nan_to_num(valid_t, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp extreme sentinel values and zero-out invalid pixels using valid mask
        data_t = torch.where(torch.abs(data_t) > 1e4, torch.zeros_like(data_t), data_t)
        data_t = data_t * valid_t
        if data_t.shape[1] == BAND_MEAN.numel():
            mean = BAND_MEAN.view(1, -1, 1, 1)
            std = BAND_STD.view(1, -1, 1, 1)
            data_t = (data_t - mean) / std
            data_t = torch.clamp(data_t, -6.0, 6.0)
            data_t = data_t * valid_t

        return data_t, mask_t, valid_t

    # SMALL HELPER for external inspection
    def get_last_debug_info(self):
        return self.last_debug_info


# -----------------------------
# Data prep helpers
# -----------------------------
def discover_ids(data_dir: Path) -> List[str]:
    ids = []
    for p in sorted(Path(data_dir).glob("*_data.npy")):
        gid = p.stem.replace("_data", "")
        if (Path(data_dir) / f"{gid}_mask.npy").exists():
            ids.append(gid)
    return ids


def stratified_split_ids(data_dir: Path, ids: List[str], test_size: float, seed: int) -> Tuple[List[str], List[str]]:
    # Simple stratify: any positive pixel across T/H/W -> class 1
    y = []
    for gid in tqdm(ids, desc="Scanning masks for stratify label"):
        m = np.load(Path(data_dir) / f"{gid}_mask.npy")
        y.append(int((m > 0).sum() > 0))
    y = np.asarray(y, dtype=int)
    uniq, counts = np.unique(y, return_counts=True)
    can_stratify = (len(uniq) > 1) and (counts.min() >= 2)
    from sklearn.model_selection import train_test_split
    if can_stratify:
        tr, va = train_test_split(ids, test_size=test_size, stratify=y, random_state=seed, shuffle=True)
    else:
        tr, va = train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)
    return list(tr), list(va)


def make_dataloaders(
    cfg: ModelConfig,
    train_ids: List[str],
    val_ids: List[str],
    train_crops_per_epoch: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, SpatiallyRobustVeduDataset, SpatiallyRobustVeduDataset]:
    train_crops_per_epoch = cfg.train_crops_per_sample if train_crops_per_epoch is None else train_crops_per_epoch
    train_ds = SpatiallyRobustVeduDataset(
        cfg.data_dir,
        train_ids,
        training_window_size=cfg.training_window_size,
        is_training=True,
        enable_augmentation=cfg.enable_augmentation,
        crops_per_epoch=train_crops_per_epoch,
        seed=cfg.seed,
        verbose=cfg.verbose,
        debug_every_n_batches=cfg.debug_every_n_batches,
        negative_crop_prob=cfg.train_negative_crop_prob,
    )
    val_ds = SpatiallyRobustVeduDataset(
        cfg.data_dir,
        val_ids,
        training_window_size=cfg.training_window_size,
        is_training=True,
        enable_augmentation=True,
        crops_per_epoch=cfg.validation_crops_per_sample,
        seed=cfg.seed + 1,
        verbose=cfg.verbose,
        debug_every_n_batches=cfg.debug_every_n_batches,
        negative_crop_prob=cfg.validation_negative_crop_prob,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    return train_loader, val_loader, train_ds, val_ds


# -----------------------------
# Metrics
# -----------------------------
def dice_coeff(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item() + eps
    return 2.0 * inter / denom


# -----------------------------
# Training
# -----------------------------
def build_model(cfg: ModelConfig) -> nn.Module:
    model = PixelTemporalPhenology(
        input_bands=cfg.input_bands,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.attn_heads,
        ff_dim=cfg.hidden_dim * 2,
        n_layers=cfg.lstm_layers,
        dropout=cfg.lstm_dropout,
        use_checkpointing=cfg.use_checkpointing,
    )
    # NEW: bias final conv to prefer background (e.g., p0=0.01)
    p0 = 0.01
    b = math.log(p0 / (1 - p0))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and getattr(m, "out_channels", None) == 1 and m.bias is not None:
                m.bias.fill_(b)
    return model.to(cfg.device)

def train(
    cfg: ModelConfig,
    exp_name: str = "balanced_spatially_robust",
    alpha: float = 0.5,
    beta: float = 0.5,
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
    train_crops_per_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Trains a model. If train_ids/val_ids are None, they are discovered & split.
    Returns a dict with best model path and histories.
    """
    cfg.prepare()
    set_seed(cfg.seed)
    train_crops_per_epoch = cfg.train_crops_per_sample if train_crops_per_epoch is None else train_crops_per_epoch

    # --- NEW: unique per-run id & derived artifact names ---
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    base_name = f"model_{exp_name}_{run_id}"
    best_path = cfg.output_dir / f"{base_name}_best.pt"
    curve_path = cfg.output_dir / f"training_curves_{base_name}.png"
    manifest_path = cfg.output_dir / f"run_{base_name}_manifest.json"
    print(f"Run ID: {run_id} | saving model as: {best_path.name}")

    # discover IDs if not supplied
    if train_ids is None or val_ids is None:
        all_ids = discover_ids(cfg.data_dir)
        if not all_ids:
            raise ValueError(f"No *_data.npy + *_mask.npy pairs found in {cfg.data_dir}")
        train_ids, val_ids = stratified_split_ids(cfg.data_dir, all_ids, cfg.train_val_split_size, cfg.seed)

    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Device: {cfg.device}; Window: {cfg.training_window_size}x{cfg.training_window_size}")
    print(
        f"Train: {len(train_ids)} | Val: {len(val_ids)} | "
        f"Train crops/sample: {train_crops_per_epoch} | Val crops/sample: {cfg.validation_crops_per_sample}"
    )
    print("=" * 70)

    train_loader, val_loader, train_ds, val_ds = make_dataloaders(cfg, train_ids, val_ids, train_crops_per_epoch)
    model = build_model(cfg)
    # NEW: use ComboLoss (Tversky + weighted BCE)
    criterion = ComboLoss(
        alpha=0.3,   # lighter FP penalty
        beta=0.7,    # heavier FN penalty (recall focus)
        gamma=1.0,
        bce_pos=3.0, # missing positives hurt more
        bce_neg=1.0,
        mix=0.5      # balance Tversky/BCE
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    # removed: old best_path assignment (now handled above with unique name)
    train_losses, val_losses = [], []
    val_dice_hist = []
    # Track best threshold alongside best model
    best_epoch_threshold = 0.5

    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        model.train()
        run_loss, n_batches = 0.0, 0
        for data, target, valid in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} Train", leave=False):
            data, target, valid = data.to(cfg.device), target.to(cfg.device), valid.to(cfg.device)
            # Guard: skip batches with bad numerics or empty valid mask
            if not torch.isfinite(data).all() or not torch.isfinite(target).all() or not torch.isfinite(valid).all():
                print("  [train skip] non-finite values detected; skipping batch")
                continue
            if valid.sum() == 0:
                if epoch % 5 == 0:
                    print("  [train skip] valid mask empty; skipping batch")
                continue
            optimizer.zero_grad()
            logits = model(data)
            prob = torch.sigmoid(logits)
            if valid.sum() > 0:
                sparsity_term = SPARSITY_LAMBDA * (prob * valid).sum() / (valid.sum() + 1e-6)
            else:
                sparsity_term = 0.0
            loss = criterion(logits, target, valid_mask=valid) + sparsity_term
            if not torch.isfinite(loss):
                print("  [train skip] loss is non-finite; skipping batch")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print("  [train skip] grad norm is non-finite; skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            run_loss += loss.item()
            n_batches += 1
        avg_train = run_loss / max(1, n_batches)
        train_losses.append(avg_train)

        # Validation phase (+ threshold sweep)
        model.eval()
        run_val, n_bv = 0.0, 0
        val_batch_losses = []
        val_batch_dice = []
        # NEW: collect for threshold sweep
        all_probs, all_tgts, all_valids = [], [], []

        with torch.no_grad():
            for bidx, (data, target, valid) in enumerate(val_loader):
                data, target, valid = data.to(cfg.device), target.to(cfg.device), valid.to(cfg.device)
                # Guard: skip bad batches
                if not torch.isfinite(data).all() or not torch.isfinite(target).all() or not torch.isfinite(valid).all():
                    print("  [val skip] non-finite values detected; skipping batch")
                    continue
                if valid.sum() == 0:
                    if epoch % 5 == 0:
                        print("  [val skip] valid mask empty; skipping batch")
                    continue

                logits = model(data)

                # loss (no sparsity in val)
                vloss = criterion(logits, target, valid_mask=valid)
                if not torch.isfinite(vloss):
                    print("  [val skip] loss is non-finite; skipping batch")
                    continue
                val_batch_losses.append(vloss.item())
                run_val += vloss.item()
                n_bv += 1

                prob = torch.sigmoid(logits)
                # collect for threshold sweep
                all_probs.append(prob.detach().cpu().flatten())
                all_tgts.append(target.detach().cpu().flatten())
                all_valids.append(valid.detach().cpu().flatten())  # NEW: collect valid mask

                # Dice at 0.5 (reporting)
                pred05 = (prob >= 0.5).float()
                d = dice_coeff(pred05, target, valid_mask=valid)
                val_batch_dice.append(d)

                if bidx < 3 and epoch < 3:
                    print(
                        f"  [val dbg] epoch {epoch} b{bidx}: "
                        f"loss={vloss.item():.6f}, dice@0.5={d:.4f}, "
                        f"prob=[{prob.min().item():.3f},{prob.max().item():.3f}], "
                        f"target_sum={target.sum().item()}"
                    )
                if bidx == 0:  # calibration snapshot
                    avg_p = prob.mean().item()
                    frac_over_05 = (prob >= 0.5).float().mean().item()
                    print(f"    [val stats] avg_p={avg_p:.3f}, frac>0.5={frac_over_05:.3f}")

        avg_val = run_val / max(1, n_bv)
        val_losses.append(avg_val)

        # NEW: threshold sweep to maximize Dice on validation
        if all_probs:
            probs_cat = torch.cat(all_probs)
            tgts_cat = torch.cat(all_tgts)
            vmask_cat = torch.cat(all_valids)
            keep = vmask_cat > 0.5
            if keep.any():
                probs_cat = probs_cat[keep]
                tgts_cat  = tgts_cat[keep]
                ths = torch.linspace(0.02, 0.98, steps=25)
                epoch_best_thresh, epoch_best_dice = 0.5, 0.0
                for t in ths:
                    pred = (probs_cat >= t).float()
                    inter = (pred * tgts_cat).sum().item()
                    denom = pred.sum().item() + tgts_cat.sum().item() + 1e-6
                    d = 2.0 * inter / denom
                    if d > epoch_best_dice:
                        epoch_best_dice, epoch_best_thresh = float(d), float(t)
            else:
                epoch_best_thresh, epoch_best_dice = 0.5, 0.0
            print(f"  [val] epoch {epoch+1}: best_thresh={epoch_best_thresh:.2f}, best_dice={epoch_best_dice:.4f}")
        else:
            epoch_best_thresh, epoch_best_dice = 0.5, 0.0

        mean_val_dice = float(np.mean(val_batch_dice)) if val_batch_dice else 0.0
        val_dice_hist.append(mean_val_dice)

        # Save best model (by val loss) and remember its threshold
        if avg_val < best_val:
            best_val = avg_val
            best_epoch_threshold = epoch_best_thresh  # NEW: keep threshold with best model
            torch.save(model.state_dict(), best_path)

        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1:02d} | train {avg_train:.4f} | val {avg_val:.4f} | "
                f"val_dice@0.5 {mean_val_dice:.4f} | best {best_val:.4f}")

    # plot curves (use unique curve_path)
    plt.figure(figsize=(9, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_dice_hist, label="Val Dice")
    plt.xlabel("Epoch"); plt.grid(True); plt.legend()
    plt.title(f"Training Curves - {exp_name}")
    plt.savefig(curve_path, dpi=150); plt.close()

    # Save run manifest with run_id, unique model path, and best threshold
    manifest = {
        "exp_name": exp_name,
        "run_id": run_id,
        "best_val_loss": float(best_val),
        "best_model_path": str(best_path),
        # NEW: persist best threshold for this trained checkpoint
        "best_threshold": float(best_epoch_threshold),
        "training_curves_path": str(curve_path),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "config": cfg.to_jsonable(),
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
    }
    with open(manifest_path, "w") as f:
        json.dump(make_json_safe(manifest), f, indent=2, default=_json_default)

    print(f"✅ Done. Run ID: {run_id} | Best val loss={best_val:.4f}. "
          f"Saved: {best_path.name}, {curve_path.name} | best_thresh={best_epoch_threshold:.2f}")
    return manifest


# -----------------------------
# Evaluation / Testing
# -----------------------------

# NEW: helper to find and load the manifest matching a model checkpoint
def _find_manifest_for_model(model_path: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        # First try a direct name guess
        guess = output_dir / f"run_{model_path.stem.replace('_best', '')}_manifest.json"
        if guess.exists():
            with open(guess, "r") as f:
                return json.load(f)

        # Fallback: scan manifests and match by best_model_path filename
        for m in sorted(output_dir.glob("run_*_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                obj = json.loads(m.read_text())
                bm = obj.get("best_model_path")
                if bm:
                    # match by full path or by filename
                    if str(model_path) == bm or Path(bm).name == Path(model_path).name:
                        return obj
            except Exception:
                continue
    except Exception:
        pass
    return None

@torch.no_grad()
def evaluate(
    cfg: ModelConfig,
    model_path: Path,
    test_ids: List[str],
    threshold: Optional[float] = None,
    crops_per_sample: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a separate directory / ID list.
    If threshold is None, uses 0.5.
    """

    cfg.prepare()
    set_seed(cfg.seed + 123)

    if threshold is None:
        threshold = 0.5

    print(f"[eval] Loading model from {model_path}")
    model = build_model(cfg)
    state = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    test_ds = SpatiallyRobustVeduDataset(
        cfg.data_dir,
        test_ids,
        training_window_size=cfg.training_window_size,
        is_training=False,
        enable_augmentation=False,
        crops_per_epoch=crops_per_sample,
        seed=cfg.seed + 999,
        verbose=False,
        debug_every_n_batches=999999,
        negative_crop_prob=cfg.eval_negative_crop_prob,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    dice_scores = []
    all_losses = []
    criterion = ComboLoss(
        alpha=0.3,
        beta=0.7,
        gamma=1.0,
        bce_pos=3.0,
        bce_neg=1.0,
        mix=0.5,
    )

    with torch.no_grad():
        for data, mask, valid in tqdm(test_loader, desc="Eval"):
            data = data.to(cfg.device)
            mask = mask.to(cfg.device)
            valid = valid.to(cfg.device)

            if not torch.isfinite(data).all() or not torch.isfinite(mask).all() or not torch.isfinite(valid).all():
                print("  [eval skip] non-finite values detected; skipping batch")
                continue
            if valid.sum() == 0:
                print("  [eval skip] valid mask empty; skipping batch")
                continue

            logits = model(data)
            loss = criterion(logits, mask, valid_mask=valid)
            if not torch.isfinite(loss):
                print("  [eval skip] loss is non-finite; skipping batch")
                continue
            all_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            pred = (probs >= threshold).float()
            d = dice_coeff(pred, mask, valid_mask=valid)
            dice_scores.append(d)

    results = {
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "std_dice": float(np.std(dice_scores)) if dice_scores else 0.0,
        "n_batches": len(dice_scores),
        "mean_loss": float(np.mean(all_losses)) if all_losses else 0.0,
        "threshold": float(threshold),
        "test_ids": test_ids,
    }

    print(
        f"[eval] mean dice={results['mean_dice']:.4f} ± {results['std_dice']:.4f} "
        f"(n_batches={results['n_batches']}) | mean loss={results['mean_loss']:.6f}"
    )

    return results



# -----------------------------
# Minimal CLI (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Evaluate Cheatgrass model")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--input_bands", type=int, default=6)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ids", type=str, nargs="*", default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    cfg = ModelConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        input_bands=args.input_bands,
        training_window_size=args.window,
        batch_size=args.batch,
        seed=args.seed,
    )

    if args.mode == "train":
        train(cfg, exp_name="cli_run", alpha=args.alpha, beta=args.beta)
    else:
        if args.model_path is None or args.test_ids is None:
            raise SystemExit("--mode eval requires --model_path and --test_ids ...")
        evaluate(cfg, Path(args.model_path), test_ids=args.test_ids)
