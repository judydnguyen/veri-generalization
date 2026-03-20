"""
Show example images from the three GTSRB sources used in property generation:
  Row 1 — GTSRB test set        (in-distribution)
  Row 2 — BTSD (BTSC)           (structural OOD, different country)
  Row 3 — GTSRB + PixMix        (corruption OOD)

Usage:
    python plot_gtsrb_sources.py
    python plot_gtsrb_sources.py --n 8 --out my_fig.png
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision
from torchvision import transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
GTSRB_DIR = BASE / "datasets" / "GTSRB"
BTSD_DIR  = BASE / "datasets" / "BTSC" / "Testing"
PIXMIX_DIR = BASE / "gtsrb_pixmix_ood"

RESIZE = transforms.Resize((64, 64))

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_gtsrb_samples(n):
    """Sample n images from the GTSRB torchvision test split."""
    ds = torchvision.datasets.GTSRB(
        str(GTSRB_DIR), split="test", download=False,
        transform=transforms.ToTensor()
    )
    indices = random.sample(range(len(ds)), n)
    imgs, labels = [], []
    for i in indices:
        img, label = ds[i]
        imgs.append(transforms.ToPILImage()(RESIZE(img)))
        labels.append(label)
    return imgs, labels


def load_btsd_samples(n):
    """Sample n images from the BTSD (BTSC) Testing directory."""
    all_imgs = []
    for class_dir in sorted(BTSD_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in (".ppm", ".png", ".jpg"):
                all_imgs.append(f)
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    imgs = [RESIZE(Image.open(p).convert("RGB")) for p in chosen]
    return imgs


def load_pixmix_samples(n):
    """Sample n images from gtsrb_pixmix_ood/."""
    all_imgs = []
    for class_dir in sorted(PIXMIX_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".ppm"):
                all_imgs.append(f)
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    imgs = [RESIZE(Image.open(p).convert("RGB")) for p in chosen]
    return imgs


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(n=6, out=None, seed=42):
    random.seed(seed)

    gtsrb_imgs, gtsrb_labels = load_gtsrb_samples(n)
    btsd_imgs                = load_btsd_samples(n)
    pixmix_imgs              = load_pixmix_samples(n)

    rows = [
        ("GTSRB test set\n(in-distribution)",  gtsrb_imgs,  "#3498db"),
        ("BTSD\n(structural OOD)",             btsd_imgs,   "#e67e22"),
        ("GTSRB + PixMix\n(corruption OOD)",   pixmix_imgs, "#9b59b6"),
    ]

    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
    fig.patch.set_facecolor("#1a1a2e")

    for r, (title, imgs, color) in enumerate(rows):
        for c, img in enumerate(imgs[:n]):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
            # label GTSRB class id on top row
            if r == 0:
                ax.set_title(f"cls {gtsrb_labels[c]}", fontsize=8,
                             color="white", pad=3)

    fig.suptitle("Three GTSRB Image Sources Used in Verification",
                 fontsize=14, color="white", fontweight="bold", y=1.01)

    patches = [mpatches.Patch(color=c, label=lbl)
               for lbl, _, c in [(t.replace("\n", " "), None, col)
                                 for t, _, col in rows]]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               frameon=False, fontsize=10,
               labelcolor="white", bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()

    # Row labels via fig.text (set_ylabel is hidden by axis("off"))
    for r, (title, _, color) in enumerate(rows):
        pos = axes[r, 0].get_position()
        y_center = pos.y0 + pos.height / 2
        fig.text(pos.x0 - 0.02, y_center, title,
                 fontsize=11, color=color, fontweight="bold",
                 va="center", ha="right")

    out_path = out or str(BASE / "gtsrb_sources_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=6,    help="Number of examples per row")
    parser.add_argument("--out",  type=str, default=None, help="Output file path")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed")
    args = parser.parse_args()
    make_plot(n=args.n, out=args.out, seed=args.seed)
