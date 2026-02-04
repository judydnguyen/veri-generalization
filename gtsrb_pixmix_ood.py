#!/usr/bin/env python3
"""
Generate PixMix OOD variants for GTSRB using PixMix (add/multiply) logic
from the official PixMix CIFAR-10 implementation (Google/Apache-2 style ops).
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 32


# ============================================================
# Augment ops (from PixMix CIFAR logic, adapted to RGB 32x32)
# ============================================================

def int_parameter(level, maxval):
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)

def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)

def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)

def translate_x(pil_img, level):
    # For GTSRB, keep translation smaller than CIFAR default (IMAGE_SIZE/3 is too big)
    level = int_parameter(sample_level(level), IMAGE_SIZE / 6)  # ~5px max
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)

def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 6)  # ~5px max
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)

def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)

def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


def apply_op(pil_img, op, severity):
    return op(pil_img, severity)


# ============================================================
# Mixing ops (official PixMix add/multiply)
# ============================================================

def _clip(img):
    return torch.clamp(img, 0.0, 1.0)

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b

def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2

def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clamp(1e-37) ** b)
    return out / 2

mixings = [add, multiply]


# ============================================================
# PixMix transform (core algorithm)
# ============================================================

class PixMixTransform:
    """
    Official PixMix-style transform:
      out = img
      for i in 1..k:
        aug = Augment(img or out)
        mix_img = random fractal/feature image
        out = mixing(out, aug_or_mix_img)
    """

    def __init__(self, image_path, k=4, beta=3, severity=3, enable_mixing=True):
        self.image_path = image_path
        self.k = k
        self.beta = beta
        self.severity = severity
        self.enable_mixing = enable_mixing

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        self.mixing_images = self._load_mixing_images()
        print(f"Loaded {len(self.mixing_images)} mixing images from {image_path}")

    def _load_mixing_images(self):
        images = []
        if self.image_path is None or not os.path.exists(self.image_path):
            return images

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [
            f for f in os.listdir(self.image_path)
            if os.path.splitext(f.lower())[1] in exts
        ]

        for f in files:
            try:
                p = os.path.join(self.image_path, f)
                img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                images.append(self.to_tensor(img))
            except Exception:
                continue
        return images

    def _random_augment(self, img_tensor):
        # apply one random op with severity
        pil = self.to_pil(img_tensor)
        op = random.choice(augmentations_all)
        aug = apply_op(pil, op, self.severity)
        return self.to_tensor(aug)

    def __call__(self, img_tensor):
        # img_tensor expected in [0,1], shape [3,32,32]
        out = img_tensor.clone()

        # Always apply at least one augmentation (OOD even without mixing)
        out = self._random_augment(out)

        if not self.enable_mixing or len(self.mixing_images) == 0:
            return _clip(out)

        for _ in range(self.k):
            # In official PixMix, they sometimes augment current image or base image;
            # this variant augments current (works well and is simpler).
            aug = self._random_augment(out)

            mix_img = random.choice(self.mixing_images)
            mixing_op = random.choice(mixings)

            # 50/50: mix with aug or mix with mixing image (matches PixMix spirit)
            if np.random.random() < 0.5:
                out = mixing_op(out, aug, self.beta)
            else:
                out = mixing_op(out, mix_img, self.beta)

            out = _clip(out)

        return out


# ============================================================
# Dataset generation (class-conditional saving)
# ============================================================

def main(gtsrb_root, save_root, image_path, split, max_per_class,
         enable_mixing, k, beta, severity, seed):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.GTSRB(
        root=gtsrb_root,
        split=split,          # "train" or "test"
        download=False,
        transform=base_transform,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    pixmix_tfm = PixMixTransform(
        image_path=image_path,
        k=k,
        beta=beta,
        severity=severity,
        enable_mixing=enable_mixing,
    )

    num_classes = 43
    counters = {c: 0 for c in range(num_classes)}
    for c in range(num_classes):
        os.makedirs(os.path.join(save_root, str(c)), exist_ok=True)

    mode = "PixMix" if enable_mixing else "Augmented-only"
    print(f"Generating {mode} GTSRB OOD samples from split='{split}'")
    print(f"save_root={save_root}, max_per_class={max_per_class}, k={k}, beta={beta}, severity={severity}")

    for imgs, labels in loader:
        for i in range(imgs.size(0)):
            y = int(labels[i].item())
            if counters[y] >= max_per_class:
                continue

            x = imgs[i]
            ood = pixmix_tfm(x)

            out_path = os.path.join(save_root, str(y), f"ood_{counters[y]:06d}.png")
            transforms.ToPILImage()(ood).save(out_path)
            counters[y] += 1

        if all(counters[c] >= max_per_class for c in counters):
            break

    print("Done.")
    for c in range(num_classes):
        print(f"Class {c}: {counters[c]} images saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate GTSRB PixMix OOD dataset (official-style mixings)")
    parser.add_argument("--gtsrb-root", type=str, required=True)
    parser.add_argument("--save-root", type=str, default="gtsrb_pixmix_ood")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to PixMix source images (fractals / feature visualizations)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-per-class", type=int, default=500)

    parser.add_argument("--enable-mixing", action="store_true",
                        help="If set, enable PixMix mixing; otherwise only random augmentations.")
    parser.add_argument("--k", type=int, default=4, help="Number of PixMix mixing steps")
    parser.add_argument("--beta", type=float, default=5.0, help="Beta for mixing coefficients")
    parser.add_argument("--severity", type=int, default=3, help="Augmentation severity (1-10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (set -1 for none)")

    args = parser.parse_args()
    seed = None if args.seed == -1 else args.seed

    main(
        gtsrb_root=args.gtsrb_root,
        save_root=args.save_root,
        image_path=args.image_path,
        split=args.split,
        max_per_class=args.max_per_class,
        enable_mixing=args.enable_mixing,
        k=args.k,
        beta=args.beta,
        severity=args.severity,
        seed=seed,
    )
