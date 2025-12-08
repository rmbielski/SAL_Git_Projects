# --- INTERACTIVE TENSOR VIEWER: browse a random sample & all time slices ---
# Usage: from tensor_viewer import launch_tensor_viewer; launch_tensor_viewer("/path/to/data_dir")
# Data dir must contain *_data.npy / *_mask.npy (optional *_valid.npy)

import os
import random
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

DEFAULT_DIR = Path("cheatgrass_data")


def _choose_rgb_indices(names):
    wants = ["B04", "B03", "B02"]
    idx = [names.index(n) for n in wants if n in names]
    if len(idx) != 3:
        idx = list(range(min(3, len(names))))
    while len(idx) < 3 and idx:
        idx.append(idx[-1])
    return idx if idx else [0, 0, 0]


def _stretch(img, lo, hi):
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def _load_sample(data_dir: Path, gid: str, valid_suffix: bool = True):
    d = np.load(data_dir / f"{gid}_data.npy")  # (T,H,W,C)
    m = np.load(data_dir / f"{gid}_mask.npy")  # (T,H,W) or (H,W)
    v = None
    if valid_suffix:
        vp = data_dir / f"{gid}_valid.npy"
        if vp.exists():
            v = np.load(vp)
    T, H, W, C = d.shape
    flat = d.reshape(-1, C)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    names = (
        ["B02", "B03", "B04", "B05", "B07", "B11"][:C]
        if C >= 3
        else [f"B{i:02d}" for i in range(1, C + 1)]
    )
    return {"gid": gid, "data": d, "mask": m, "valid": v, "mn": mn, "mx": mx, "names": names}


def launch_tensor_viewer(data_dir: Path | str | None = None):
    """Start interactive viewer in current notebook cell."""
    data_dir = Path(data_dir or os.getenv("CHEATGRASS_TENSOR_DIR", DEFAULT_DIR))
    ids_data = {p.name[:-9] for p in data_dir.glob("*_data.npy")}
    ids_mask = {p.name[:-9] for p in data_dir.glob("*_mask.npy")}
    ids = sorted(ids_data & ids_mask)

    if not ids:
        print(f"No tensors found in {data_dir} (expected *_data.npy and *_mask.npy).")
        return

    state = {}

    sample = W.Dropdown(options=ids, value=random.choice(ids), description="Sample")
    t_slider = W.IntSlider(min=0, max=1, value=0, step=1, description="Time")
    stretch_mode = W.Dropdown(
        options=[("Per-slice", "slice"), ("Per-band global", "global")],
        value="slice",
        description="Stretch",
    )
    overlay = W.Checkbox(value=True, description="Overlay mask")
    overlay_valid = W.Checkbox(value=True, description="Overlay valid", indent=False)
    pick_btn = W.Button(description="Random", tooltip="Pick a random sample")
    out = W.Output()

    def refresh_state(gid: str):
        s = _load_sample(data_dir, gid)
        state.clear()
        state.update(s)
        t_slider.max = s["data"].shape[0] - 1
        t_slider.value = min(t_slider.value, t_slider.max)

    def on_pick(_):
        sample.value = random.choice(ids)

    def render(*_):
        if not state:
            refresh_state(sample.value)
        d = state["data"]
        m = state["mask"]
        v = state.get("valid")
        names = state["names"]
        mn = state["mn"]
        mx = state["mx"]
        t = t_slider.value
        C = d.shape[3]
        rgb_idx = _choose_rgb_indices(names)
        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            r_lo = mn[rgb_idx[0]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[0]].min()
            r_hi = mx[rgb_idx[0]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[0]].max()
            g_lo = mn[rgb_idx[1]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[1]].min()
            g_hi = mx[rgb_idx[1]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[1]].max()
            b_lo = mn[rgb_idx[2]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[2]].min()
            b_hi = mx[rgb_idx[2]] if stretch_mode.value == "global" else d[t, :, :, rgb_idx[2]].max()
            r = _stretch(d[t, :, :, rgb_idx[0]], r_lo, r_hi)
            g = _stretch(d[t, :, :, rgb_idx[1]], g_lo, g_hi)
            b = _stretch(d[t, :, :, rgb_idx[2]], b_lo, b_hi)
            rgb = np.dstack([r, g, b])
            ax_rgb = axes[0, 0]
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"RGB {names[rgb_idx[0]]}/{names[rgb_idx[1]]}/{names[rgb_idx[2]]}")
            ax_rgb.axis("off")
            if overlay.value:
                ax_rgb.imshow(m[t], cmap="Reds", alpha=0.25)
            if overlay_valid.value and v is not None:
                ax_rgb.imshow(v[t], cmap="Greens", alpha=0.25)

            axm = axes[0, 1]
            axm.imshow(m[t], cmap="gray", vmin=0, vmax=1)
            axm.set_title("Mask")
            axm.axis("off")

            if v is not None:
                axv = axes[0, 2]
                axv.imshow(v[t], cmap="gray", vmin=0, vmax=1)
                axv.set_title("Valid mask")
                axv.axis("off")

            slots = [(0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
            slot_idx = 0
            for bi in range(min(C, 5)):
                rix, cix = slots[slot_idx]
                slot_idx += 1
                axb = axes[rix, cix]
                lo = mn[bi] if stretch_mode.value == "global" else d[t, :, :, bi].min()
                hi = mx[bi] if stretch_mode.value == "global" else d[t, :, :, bi].max()
                axb.imshow(_stretch(d[t, :, :, bi], lo, hi), cmap="viridis")
                axb.set_title(names[bi])
                axb.axis("off")

            for i in range(2):
                for j in range(4):
                    if not axes[i, j].has_data():
                        axes[i, j].axis("off")

            valid_txt = ""
            if v is not None:
                valid_txt = f" | valid frac t={v[t].mean():.2f}"
            fig.suptitle(f"{state['gid']} | data {d.shape} mask {m.shape} | t={t}{valid_txt}", fontsize=11)
            plt.tight_layout()
            plt.show()

    def on_sample_change(change):
        if change.get("name") == "value":
            refresh_state(change["new"])
            render()

    pick_btn.on_click(on_pick)
    sample.observe(on_sample_change, names="value")
    for w in (t_slider, stretch_mode, overlay, overlay_valid):
        w.observe(render, names="value")

    refresh_state(sample.value)
    ui = W.HBox([sample, pick_btn, t_slider, stretch_mode, overlay, overlay_valid])
    display(ui)
    display(out)
    render()


if __name__ == "__main__":
    launch_tensor_viewer()
