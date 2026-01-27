# cheatgrass_viz.py
"""
Visualization utilities for cheatgrass CNN evaluation.

Usage from notebook:

    from pathlib import Path
    from cheatgrass_viz import visualize_eval_and_samples

    results = evaluate(...)

    visualize_eval_and_samples(
        cfg=cfg_eval,
        best_model_path=best_model_path,
        best_threshold=best_threshold,
        test_ids=test_ids,
        eval_results=results,   # optional; will also look for eval_<model>.json
        max_examples=6,
    )
"""

from pathlib import Path
import json
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from cheatgrass_model import (
    SpatiallyRobustVeduDataset,
    build_model,
    discover_ids,
    ModelConfig,
)


def _load_eval_results(
    output_dir: Path,
    best_model_path: Path,
    eval_results: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Try to load evaluation results from JSON; fall back to in-memory dict.
    """
    if eval_results is not None:
        return eval_results

    eval_json_path = output_dir / f"eval_{best_model_path.stem}.json"
    if eval_json_path.exists():
        return json.loads(eval_json_path.read_text())

    return None


def _plot_per_batch_dice(eval_res: Dict[str, Any]) -> None:
    """
    If per_batch_dice exists, show histogram + series plots.
    Otherwise, print summary only.
    """
    if eval_res is None:
        print("No evaluation results found for per-batch plots.")
        return

    mean_d = eval_res.get("mean_dice", None)
    std_d = eval_res.get("std_dice", None)
    n_batches = eval_res.get("n_batches", None)
    print(f"Eval summary: n_batches={n_batches}, mean_dice={mean_d}, std_dice={std_d}")

    per_dice = eval_res.get("per_batch_dice", None)
    if per_dice is None:
        print("No 'per_batch_dice' field in eval results; skipping per-batch plots.")
        return

    per_dice = np.asarray(per_dice, dtype=float)
    if per_dice.size == 0:
        print("Empty 'per_batch_dice' array; skipping per-batch plots.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(per_dice, bins=20, edgecolor="k")
    plt.title("Per-batch Dice Distribution")
    plt.xlabel("Dice")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(per_dice, "o-", ms=4, alpha=0.75)
    plt.title("Per-batch Dice (by batch index)")
    plt.xlabel("Batch index")
    plt.ylabel("Dice")
    plt.grid(True)
    plt.show()


def _load_latest_detailed_csv(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load latest *_detailed_results.csv if present.
    """
    csvs = sorted(
        output_dir.glob("*_detailed_results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    print(f"Loaded detailed CSV: {csvs[0].name} (n={len(df)})")
    return df


def _summarize_csv(df: pd.DataFrame) -> None:
    """
    Print quick summary of top/bottom samples by dice (if available).
    """
    if df is None:
        print("No detailed CSV; skipping CSV summary.")
        return

    if "dice" in df.columns:
        topN = df.nlargest(6, "dice")
        botN = df.nsmallest(6, "dice")
    else:
        topN = df.head(6)
        botN = df.tail(6)

    cols = [
        c
        for c in [
            "sample_idx",
            "gid",
            "dice",
            "precision",
            "recall",
            "gt_pixels",
            "pred_pixels",
        ]
        if c in df.columns
    ]

    print("\nTop 6 samples by Dice (CSV):")
    print(topN[cols].to_string(index=False))

    print("\nBottom 6 samples by Dice (CSV):")
    print(botN[cols].to_string(index=False))


def _choose_ids_to_visualize(
    cfg: ModelConfig,
    test_ids: Optional[List[str]],
    df: Optional[pd.DataFrame],
    max_examples: int = 6,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Decide which GIDs to visualize:
      - If CSV has 'gid' + 'dice': use top/bottom by dice.
      - Else: random subset of test_ids or discover_ids(cfg.data_dir).
    """
    ids_to_vis: List[str] = []

    # Prefer CSV if it has gid + dice
    if df is not None and "gid" in df.columns and "dice" in df.columns:
        top_gids = df.nlargest(min(3, len(df)), "dice")["gid"].astype(str).tolist()
        bot_gids = df.nsmallest(min(3, len(df)), "dice")["gid"].astype(str).tolist()
        ids_to_vis = list(dict.fromkeys(top_gids + bot_gids))  # ordered unique
        return ids_to_vis[:max_examples]

    # Fallback: use provided test_ids or discover_ids
    if test_ids is None:
        all_ids = discover_ids(Path(cfg.data_dir))
    else:
        all_ids = list(test_ids)

    if not all_ids:
        return []

    rng = np.random.default_rng(seed)
    k = min(max_examples, len(all_ids))
    choice = rng.choice(all_ids, size=k, replace=False).tolist()
    return choice


def _visualize_samples(
    cfg: ModelConfig,
    best_model_path: Path,
    best_threshold: float,
    ids_to_vis: List[str],
    crop_seed: Optional[int] = None,
) -> None:
    """
    Build a tiny dataset on selected IDs and show predictions vs GT for mid time slice.
    """
    if not ids_to_vis:
        print("No IDs selected for visualization.")
        return

    print("Visualizing IDs:", ids_to_vis)

    ds_vis = SpatiallyRobustVeduDataset(
        data_dir=Path(cfg.data_dir),
        location_ids=ids_to_vis,
        training_window_size=cfg.training_window_size,
        is_training=False,
        enable_augmentation=False,
        crops_per_epoch=1,
        seed=123 if crop_seed is None else int(crop_seed),
        verbose=False,
        negative_crop_prob=0.0,  # always bias toward vegetation for viz
        min_valid_fraction=cfg.min_valid_fraction,
        add_valid_channel=cfg.add_valid_channel,
    )

    loader = torch.utils.data.DataLoader(
        ds_vis,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Ensure cfg has input_bands etc.
    cfg.prepare()
    model = build_model(cfg)
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    model.eval()

    examples = []
    with torch.no_grad():
        for idx, (data, target, valid) in enumerate(loader):
            data = data.to(cfg.device)      # (1,T,C,H,W)
            target = target.to(cfg.device)  # (1,T,1,H,W)
            valid = valid.to(cfg.device)    # (1,T,1,H,W)

            logits = model(data)
            prob = torch.sigmoid(logits)
            pred = (prob >= best_threshold).float()

            examples.append(
                dict(
                    gid=ids_to_vis[idx],
                    data=data.cpu().numpy()[0],
                    target=target.cpu().numpy()[0],
                    prob=prob.cpu().numpy()[0],
                    pred=pred.cpu().numpy()[0],
                    valid=valid.cpu().numpy()[0],
                )
            )

    if not examples:
        print("No examples collected; check dataset / IDs list.")
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)

    for i, sample in enumerate(examples):
        data = sample["data"]      # (T,C,H,W)
        target = sample["target"]  # (T,1,H,W)
        prob = sample["prob"]      # (T,1,H,W)
        pred = sample["pred"]      # (T,1,H,W)
        valid = sample["valid"]    # (T,1,H,W)

        T, C, H, W = data.shape
        # Pick the time slice with the most GT pixels; fall back to mid if all-zero
        mask_sums = target[:, 0].sum(axis=(1, 2))
        if mask_sums.max() > 0:
            display_t = int(mask_sums.argmax())
        else:
            display_t = T // 2
        mask_pix = int(mask_sums[display_t])

        rgb_bands = [2, 1, 0] if C >= 3 else [0, 0, 0]

        rgb = data[display_t, rgb_bands].transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[i, 0].imshow(rgb)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"ID {sample['gid']} - T={display_t} RGB")

        axes[i, 1].imshow(target[display_t, 0], cmap="Greens", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"GT mask (pix={mask_pix})")

        im = axes[i, 2].imshow(prob[display_t, 0], cmap="Reds", vmin=0, vmax=1)
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Prob (cheatgrass)")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        axes[i, 3].imshow(pred[display_t, 0], cmap="Blues", vmin=0, vmax=1)
        axes[i, 3].axis("off")
        axes[i, 3].set_title(f"Pred (thr={best_threshold:.2f})")

    plt.tight_layout()
    plt.show()


def visualize_eval_and_samples(
    cfg: ModelConfig,
    best_model_path: Path,
    best_threshold: float,
    test_ids: Optional[List[str]] = None,
    eval_results: Optional[Dict[str, Any]] = None,
    max_examples: int = 6,
    random_seed: Optional[int] = None,
) -> None:
    """
    High-level helper:
      - Summarize eval JSON or provided eval_results dict.
      - Plot per-batch Dice if available.
      - Summarize detailed CSV (if any).
      - Visualize a few sample predictions.

    Parameters
    ----------
    cfg : ModelConfig
        Config for the dataset (usually cfg_eval).
    best_model_path : Path
        Path to the best model checkpoint (from manifest).
    best_threshold : float
        Threshold used to binarize probabilities.
    test_ids : list[str], optional
        IDs in the test set. If None, discover_ids(cfg.data_dir) is used.
    eval_results : dict, optional
        In-memory results from evaluate(); if None, tries eval_<model>.json.
    max_examples : int
        Max number of samples to visualize.
    """
    output_dir = Path(cfg.output_dir)

    print(f"=== Visualization ===")
    print(f"output_dir:      {output_dir}")
    print(f"best_model_path: {best_model_path}")
    print(f"best_threshold:  {best_threshold}")
    print(f"data_dir:        {cfg.data_dir}\n")

    # 1) Eval summary + per-batch plots
    eval_res = _load_eval_results(output_dir, best_model_path, eval_results)
    _plot_per_batch_dice(eval_res)

    # 2) Detailed CSV summary (optional)
    df = _load_latest_detailed_csv(output_dir)
    _summarize_csv(df)

    # 3) Sample-level visualizations
    ids_to_vis = _choose_ids_to_visualize(
        cfg,
        test_ids,
        df,
        max_examples=max_examples,
        seed=random_seed,
    )
    _visualize_samples(cfg, best_model_path, best_threshold, ids_to_vis, crop_seed=random_seed)
