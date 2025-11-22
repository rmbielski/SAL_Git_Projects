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
    hidden_dim: int = 128
    input_bands: int = 6
    train_val_split_size: float = 0.2
    training_window_size: int = 32
    enable_augmentation: bool = True
    validation_crops_per_sample: int = 3
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reproducibility
    seed: int = 42
    # dataloader
    num_workers: int = 0
    pin_memory: bool = False

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


def enhanced_random_crop_sample(
    data: torch.Tensor,  # (T, C, H, W)
    mask: torch.Tensor,  # (T, 1, H, W) or (1, 1, H, W)
    crop_size: int,
    is_training: bool = True,
    force_spatial_diversity: bool = True,
    debug: bool = False,   # NEW: optional debug return
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, int, bool, int]]]:
    t, c, h, w = data.shape
    if h == crop_size and w == crop_size:
        if debug:
            # no augmentation needed: returns center coords & mask sum
            mask_sum = int(mask.sum().item())
            return data, mask, ((h - crop_size)//2, (w - crop_size)//2, False, mask_sum)
        return data, mask, None

    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        data = torch.nn.functional.pad(data, (0, pad_w, 0, pad_h))
        mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
        _, _, h, w = data.shape

    if (is_training and force_spatial_diversity) or (h > crop_size and w > crop_size):
        first_mask = mask[0, 0].detach().cpu().numpy()
        top, left = find_valid_crop_position_enhanced(first_mask, crop_size, force_include_vegetation=True)
        augmented = True
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        augmented = False

    dc = data[:, :, top : top + crop_size, left : left + crop_size]
    mc = mask[:, :, top : top + crop_size, left : left + crop_size]

    if debug:
        mask_sum = int(mc.sum().item())
        return dc, mc, (top, left, bool(augmented), mask_sum)
    return dc, mc, None


# -----------------------------
# Model
# -----------------------------
class PhenologyAwareUNet(nn.Module):
    """
    Sequence model: per-time-step 2D CNN -> sequence LSTM -> per-step decoder.
    Expects spatial crop 32x32.
    """
    def __init__(self, input_bands: int = 6, hidden_dim: int = 128, output_classes: int = 1):
        super().__init__()
        self.feature_cnn = nn.Sequential(
            nn.Conv2d(input_bands, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(True),  # 32 -> 30
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(True),          # 30 -> 28
            nn.MaxPool2d(kernel_size=2, stride=2),                                         # 28 -> 14
        )
        fs = 64 * 14 * 14  # flattened feature per-time-step
        self.lstm = nn.LSTM(fs, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, fs)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0),  # 28 -> 32
            nn.ReLU(True),
            nn.Conv2d(32, output_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), H=W=32
        b, t, c, h, w = x.shape
        if (h, w) != (32, 32):
            raise ValueError(f"Model expects 32x32 input, got {h}x{w}")

        cnn_list = []
        for i in range(t):
            f = self.feature_cnn(x[:, i])  # (B, 64, 14, 14)
            cnn_list.append(f.flatten(1))  # (B, fs)

        seq = torch.stack(cnn_list, 1)  # (B, T, fs)
        lstm_out, _ = self.lstm(seq)    # (B, T, 2*hidden)

        outs = []
        for i in range(t):
            p = self.proj(lstm_out[:, i]).view(b, 64, 14, 14)
            d = self.decoder(p)  # (B, 1, 32, 32)
            outs.append(d)
        return torch.stack(outs, 1)  # (B, T, 1, 32, 32)


# -----------------------------
# Dataset
# -----------------------------
class SpatiallyRobustVeduDataset(Dataset):
    """
    Expects files:
      {id}_data.npy  -> shape (T, H, W, C)
      {id}_mask.npy  -> shape (T, H, W) or (H, W)
    Returns tensors:
      data: (T, C, 32, 32), mask: (T, 1, 32, 32)
    """
    def __init__(
        self,
        data_dir: Path,
        location_ids: List[str],
        training_window_size: int = 32,
        is_training: bool = True,
        enable_augmentation: bool = True,
        crops_per_epoch: int = 1,
        seed: int = 42,
        verbose: bool = False,
        debug_every_n_batches: int = 50,
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

        data = np.transpose(data, (0, 3, 1, 2))  # (T,C,H,W)

        if mask.ndim == 3:
            mask = np.expand_dims(mask, 1)  # (T,1,H,W)
        elif mask.ndim == 2:
            mask = mask[None, None, ...]    # (1,1,H,W)

        # Convert to tensors (data_t, mask_t)
        data_t = torch.from_numpy(data)
        mask_t = torch.from_numpy(mask)

        # Apply enhanced random cropping; return debug info if verbose
        if self.enable_augmentation or (data_t.shape[2], data_t.shape[3]) != (self.training_window_size, self.training_window_size):
            dc, mc, debug_info = enhanced_random_crop_sample(
                data_t, mask_t, self.training_window_size, is_training=self.is_training, force_spatial_diversity=self.enable_augmentation, debug=self.verbose
            )
            data_t, mask_t = dc, mc
        else:
            debug_info = None

        if self.verbose:
            gid = self.location_ids[sample_idx]
            top, left, aug_flag, mask_sum = debug_info if debug_info is not None else ((data_t.shape[2]-self.training_window_size)//2, (data_t.shape[3]-self.training_window_size)//2, False, int(mask_t.sum().item()))
            self.last_debug_info = {
                "gid": gid,
                "top": int(top),
                "left": int(left),
                "augmentation": bool(aug_flag),
                "mask_sum_in_crop": int(mask_sum),
                "crop_size": int(self.training_window_size),
            }

        return data_t, mask_t

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
    train_crops_per_epoch: int = 3,
) -> Tuple[DataLoader, DataLoader, SpatiallyRobustVeduDataset, SpatiallyRobustVeduDataset]:
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
    )
    val_ds = SpatiallyRobustVeduDataset(
        cfg.data_dir,
        val_ids,
        training_window_size=cfg.training_window_size,
        is_training=True,  # random crops for validation for robustness
        enable_augmentation=True,
        crops_per_epoch=cfg.validation_crops_per_sample,
        seed=cfg.seed + 1,
        verbose=cfg.verbose,
        debug_every_n_batches=cfg.debug_every_n_batches,
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
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    # pred/target: (B,T,1,H,W) in {0,1}
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item() + eps
    return 2.0 * inter / denom


# -----------------------------
# Training
# -----------------------------
def build_model(cfg: ModelConfig) -> nn.Module:
    model = PhenologyAwareUNet(input_bands=cfg.input_bands, hidden_dim=cfg.hidden_dim, output_classes=1)
    return model.to(cfg.device)


def train(
    cfg: ModelConfig,
    exp_name: str = "balanced_spatially_robust",
    alpha: float = 0.5,
    beta: float = 0.5,
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
    train_crops_per_epoch: int = 3,
) -> Dict[str, Any]:
    """
    Trains a model. If train_ids/val_ids are None, they are discovered & split.
    Returns a dict with best model path and histories.
    """
    cfg.prepare()
    set_seed(cfg.seed)

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
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Val crops/sample: {cfg.validation_crops_per_sample}")
    print("=" * 70)

    train_loader, val_loader, train_ds, val_ds = make_dataloaders(cfg, train_ids, val_ids, train_crops_per_epoch)
    model = build_model(cfg)
    criterion = FocalTverskyLoss(alpha=alpha, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    # removed: old best_path assignment (now handled above with unique name)
    train_losses, val_losses = [], []
    val_dice_hist = []
    VAL_THRESH = 0.5


    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        model.train()
        run_loss, n_batches = 0.0, 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} Train", leave=False):
            data = data.to(cfg.device)            # (B,T,C,32,32)
            target = target.to(cfg.device)        # (B,T,1,32,32)
            optimizer.zero_grad()
            logits = model(data)                  # (B,T,1,32,32)
            loss = criterion(logits.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            n_batches += 1
        avg_train = run_loss / max(1, n_batches)
        train_losses.append(avg_train)

        model.eval()
        run_val, n_bv = 0.0, 0
        val_batch_losses = []
        val_batch_dice = []

        with torch.no_grad():
            for bidx, (data, target) in enumerate(val_loader):
                data = data.to(cfg.device)
                target = target.to(cfg.device)
                logits = model(data)

                # loss
                loss = criterion(logits.flatten(), target.flatten())
                val_batch_losses.append(loss.item())
                run_val += loss.item()
                n_bv += 1

                # dice @ threshold
                prob = torch.sigmoid(logits)
                pred = (prob >= VAL_THRESH).float()
                d = dice_coeff(pred, target)
                val_batch_dice.append(d)

                if bidx < 3 and epoch < 3:
                    print(
                        f"  [val dbg] epoch {epoch} b{bidx}: "
                        f"loss={loss.item():.6f}, dice={d:.4f}, "
                        f"prob=[{prob.min().item():.3f},{prob.max().item():.3f}], "
                        f"target_sum={target.sum().item()}"
                    )

        avg_val = run_val / max(1, n_bv)
        val_losses.append(avg_val)

        mean_val_dice = float(np.mean(val_batch_dice)) if val_batch_dice else 0.0
        val_dice_hist.append(mean_val_dice)


        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            print(f"Epoch {epoch+1:02d} | train {avg_train:.4f} | val {avg_val:.4f} | "
                f"val_dice {mean_val_dice:.4f} | best {best_val:.4f}")


    # plot curves (use unique curve_path)
    plt.figure(figsize=(9, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_dice_hist, label="Val Dice")
    plt.xlabel("Epoch"); plt.grid(True); plt.legend()
    plt.title(f"Training Curves - {exp_name}")
    plt.savefig(curve_path, dpi=150); plt.close()

    # Save run manifest with run_id and unique model path
    manifest = {
        "exp_name": exp_name,
        "run_id": run_id,
        "best_val_loss": float(best_val),
        "best_model_path": str(best_path),
        "training_curves_path": str(curve_path),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "config": cfg.to_jsonable(),
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
    }
    with open(manifest_path, "w") as f:
        json.dump(make_json_safe(manifest), f, indent=2, default=_json_default)

    print(f"✅ Done. Run ID: {run_id} | Best val loss={best_val:.4f}. Saved: {best_path.name}, {curve_path.name}")
    return manifest


# -----------------------------
# Evaluation / Testing
# -----------------------------
@torch.no_grad()
def evaluate(
    cfg: ModelConfig,
    model_path: Path,
    test_ids: List[str],
    threshold: float = 0.5,
    crops_per_sample: int = 3,
) -> Dict[str, Any]:
    """
    Loads a saved model and evaluates on held-out IDs (random-crop robustness).
    Returns summary dict with mean Dice and per-id metrics.
    """
    cfg.prepare()
    set_seed(cfg.seed + 999)

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()

    test_ds = SpatiallyRobustVeduDataset(
        cfg.data_dir,
        test_ids,
        training_window_size=cfg.training_window_size,
        is_training=False,            # no explicit "training" aug flags—but we still use crops
        enable_augmentation=True,     # random crops to assess robustness
        crops_per_epoch=crops_per_sample,
        seed=cfg.seed + 2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    per_batch = []
    for data, target in tqdm(test_loader, desc="Testing"):
        data = data.to(cfg.device)
        target = target.to(cfg.device)
        logits = model(data)
        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).float()
        d = dice_coeff(pred, target)
        per_batch.append(d)

    mean_dice = float(np.mean(per_batch)) if per_batch else 0.0
    std_dice = float(np.std(per_batch)) if per_batch else 0.0

    results = {
        "model_path": str(model_path),
        "threshold": float(threshold),
        "mean_dice": float(mean_dice),
        "std_dice": float(std_dice),
        "per_batch_dice": [float(x) for x in per_batch],
        "n_batches": int(len(per_batch)),
        "config": cfg.to_jsonable(),
        "test_ids": [str(t) for t in test_ids],
    }

    out_json = cfg.output_dir / f"eval_{Path(model_path).stem}.json"
    # Make sure everything is JSON-serializable and then write
    safe_results = make_json_safe(results)
    with open(out_json, "w") as f:
        json.dump(safe_results, f, indent=2, default=_json_default)
    print(f"📊 Eval mean Dice={mean_dice:.4f} (±{std_dice:.4f})  -> {out_json.name}")

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
    parser.add_argument("--window", type=int, default=32)
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
