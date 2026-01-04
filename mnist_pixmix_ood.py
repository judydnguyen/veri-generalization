import os
import random
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Augmentation and mixing for MNIST OOD generation
# ============================================================

def _clip(img):
    return torch.clamp(img, 0.0, 1.0)


def _rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def pixmix(image, mixing_image, beta=3):
    lam = np.random.beta(beta, beta)
    W, H = image.shape[-2], image.shape[-1]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(W, H, lam)

    mixed = image.clone()
    mixed[:, bbx1:bbx2, bby1:bby2] = mixing_image[:, bbx1:bbx2, bby1:bby2]
    return _clip(mixed)


class Augment:
    def __init__(self):
        # Create augmentation transform pipeline
        self.augment = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ])

    def __call__(self, img):
        # Convert tensor to PIL, augment, then back to tensor
        pil_img = transforms.ToPILImage()(img)
        aug_img = self.augment(pil_img)
        return transforms.ToTensor()(aug_img)


class PixMixAugment:
    def __init__(self, beta=3, k=4, enable_mixing=True, image_path=None):
        self.beta = beta
        self.k = k
        self.enable_mixing = enable_mixing
        self.image_path = image_path or "/home/judy/code/unlearning-verification/fractals_and_fvis/first_layers_resized256_onevis/images"
        
        # Load and preprocess mixing images from path
        self.mixing_images = self._load_mixing_images()
        print(f"Loaded {len(self.mixing_images)} mixing images from {self.image_path}")

    def _load_mixing_images(self):
        """Load images from path, convert to grayscale, resize, and convert to tensors."""
        mixing_images = []
        
        if not os.path.exists(self.image_path):
            print(f"Warning: Image path does not exist: {self.image_path}")
            return mixing_images
        
        # Get all image files from the directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in os.listdir(self.image_path)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if len(image_files) == 0:
            print(f"Warning: No image files found in {self.image_path}")
            return mixing_images
        
        # Transform to convert images: grayscale, resize to 28x28 (MNIST size), then to tensor
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((28, 28)),  # Resize to MNIST dimensions
            transforms.ToTensor(),  # Convert to tensor [C, H, W] with values in [0, 1]
        ])
        
        for img_file in image_files:
            try:
                img_path = os.path.join(self.image_path, img_file)
                img = Image.open(img_path)
                
                # Convert RGBA to RGB if needed (before grayscale conversion)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Apply transforms: grayscale, resize, to tensor
                img_tensor = transform(img)
                mixing_images.append(img_tensor)
            except Exception as e:
                print(f"Warning: Failed to load {img_file}: {e}")
                continue
        
        return mixing_images

    def __call__(self, img, mixing_imgs=None):
        """
        Apply PixMix mixing.
        
        Args:
            img: Input image tensor [C, H, W]
            mixing_imgs: Optional list of mixing images (if None, uses images from path)
        
        Returns:
            Mixed image tensor
        """
        out = img
        if self.enable_mixing:
            # Use images from path if available, otherwise use provided mixing_imgs
            if len(self.mixing_images) > 0:
                mixing_pool = self.mixing_images
            elif mixing_imgs is not None and len(mixing_imgs) > 0:
                mixing_pool = mixing_imgs
            else:
                print("Warning: No mixing images available, skipping mixing")
                return out
            
            for _ in range(self.k):
                mix_img = random.choice(mixing_pool)
                # Ensure mixing image has same dimensions as input
                if mix_img.shape != out.shape:
                    # Resize mixing image to match input dimensions
                    mix_img_pil = transforms.ToPILImage()(mix_img)
                    mix_img_pil = mix_img_pil.resize((out.shape[-1], out.shape[-2]), Image.LANCZOS)
                    mix_img = transforms.ToTensor()(mix_img_pil)
                out = pixmix(out, mix_img, self.beta)
        return out


class CombinedAugment:
    """Combines standard augmentation with optional PixMix mixing."""
    def __init__(self, enable_mixing=True, beta=3, k=2, image_path=None):
        self.augment = Augment()
        self.pixmix = PixMixAugment(beta=beta, k=k, enable_mixing=enable_mixing, image_path=image_path)
        self.enable_mixing = enable_mixing

    def __call__(self, img, mixing_imgs=None):
        # Always apply standard augmentation first
        aug_img = self.augment(img)
        
        # Optionally apply mixing if enabled
        # PixMixAugment will use images from path if available, otherwise use provided mixing_imgs
        if self.enable_mixing:
            aug_img = self.pixmix(aug_img, mixing_imgs)
        
        return aug_img


# ============================================================
# MNIST OOD generation (class-conditional saving)
# ============================================================

def main(enable_mixing=False):
    """
    Generate augmented MNIST OOD samples.
    
    Args:
        enable_mixing: If True, apply PixMix mixing in addition to augmentation.
                      If False, only apply standard augmentation.
    """
    save_root = "mnist_pixmix_ood"
    batch_size = 128
    max_per_class = 500   # 🔴 control how many per class

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Image path for mixing images
    image_path = "/home/judy/code/unlearning-verification/fractals_and_fvis/first_layers_resized256_onevis/images"
    augmenter = CombinedAugment(enable_mixing=enable_mixing, beta=3, k=2, image_path=image_path)

    # create class folders
    counters = {}
    for c in range(10):
        class_dir = os.path.join(save_root, str(c))
        os.makedirs(class_dir, exist_ok=True)
        counters[c] = 0

    mode_str = "augmented + mixed" if enable_mixing else "augmented only"
    print(f"Generating class-conditional {mode_str} MNIST samples...")

    for imgs, labels in loader:
        for i in range(imgs.size(0)):
            label = labels[i].item()

            if counters[label] >= max_per_class:
                continue

            base_img = imgs[i]

            # Apply augmentation (and optionally mixing from path)
            # mixing_imgs parameter is optional - will use images from path if available
            ood_img = augmenter(base_img, mixing_imgs=None)

            pil = transforms.ToPILImage()(ood_img)
            save_path = os.path.join(
                save_root,
                str(label),
                f"ood_{counters[label]:06d}.png"
            )
            pil.save(save_path)

            counters[label] += 1

        # early stop if all classes are done
        if all(counters[c] >= max_per_class for c in counters):
            break

    print("Done.")
    for c in counters:
        print(f"Class {c}: {counters[c]} images saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented MNIST OOD samples")
    parser.add_argument(
        "--enable-mixing",
        action="store_true",
        help="Enable PixMix mixing in addition to augmentation (default: False, only augmentation)"
    )
    args = parser.parse_args()
    
    main(enable_mixing=args.enable_mixing)
