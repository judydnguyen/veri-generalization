"""
GTSRB + BTSC mapped dataset (cleaned)

- Loads subset of GTSRB
- Loads BTSC / BTSD
- Maps BTSC -> GTSRB labels
- Drops unmapped classes
- Optional compact label space
"""

import os
import csv
import math
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import onnxruntime as onnxrun


# ============================================================
# 1) BTSC → GTSRB mapping
# ============================================================

BTSC_TO_GTSRB = {
    "00002": 23,  # Slippery road
    "00009": 31,  # Wild animals crossing
    "00010": 25,  # Road work
    "00015": 24,  # Road narrows right
    "00016": 24,  # Road narrows left (mapped to "narrows right" as closest match)
    "00017": 18,  # General caution / Intersection
    "00019": 13,  # Yield
    "00021": 14,  # Stop
    "00022": 17,  # No entry
    "00023": 29,  # Bicycles prohibited (mapped to "Bicycles crossing" as related)
    "00029": 34,  # No left turn (mapped to "Turn left ahead" as related)
    "00030": 33,  # No right turn (mapped to "Turn right ahead" as related)
    "00032": 4,   # Speed limit 70
    "00035": 33,  # Turn right ahead
    # Additional speed limit mappings if available
    # Note: BTSC may have other speed limits that could map to GTSRB speed limit classes
}

BTSC_SEMANTIC = {
    "00002": "Slippery road",
    "00009": "Wild animals crossing",
    "00010": "Road work",
    "00015": "Road narrows on the right",
    "00016": "Road narrows on the left",
    "00017": "Intersection / crossroads",
    "00019": "Yield",
    "00021": "Stop",
    "00022": "No entry",
    "00023": "Bicycles prohibited",
    "00029": "No left turn",
    "00030": "No right turn",
    "00032": "Speed limit (70 km/h)",
    "00035": "Turn right ahead",
}

USED_GTSRB = sorted(set(BTSC_TO_GTSRB.values()))
GTSRB_TO_COMPACT = {c: i for i, c in enumerate(USED_GTSRB)}


# ============================================================
# 2) CSV utilities
# ============================================================

def load_btsc_csv(csv_path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing BTSC CSV: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return [(r["image_path"], r["label_name"]) for r in reader]


# ============================================================
# 3) Dataset definitions
# ============================================================

class BTSCMappedDataset(Dataset):
    def __init__(self, root: str, annotations, transform=None):
        self.samples = []
        self.transform = transform
        self.root = root

        for rel_path, btsc_id in annotations:
            if btsc_id in BTSC_TO_GTSRB:
                self.samples.append((
                    os.path.join(root, rel_path),
                    BTSC_TO_GTSRB[btsc_id]
                ))

        if not self.samples:
            raise RuntimeError("No mapped BTSC samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class GTSRBSubset(Dataset):
    def __init__(self, root: str, allowed_labels, transform=None):
        self.samples = []
        self.transform = transform

        for c in allowed_labels:
            cls_dir = os.path.join(root, f"{c:05d}")
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".ppm")):
                    self.samples.append((os.path.join(cls_dir, f), c))

        if not self.samples:
            raise RuntimeError("No GTSRB samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class CompactLabels(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, GTSRB_TO_COMPACT[label]


# Keep original labels (no compact mapping)
class OriginalLabels(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label  # Return original GTSRB label


# ============================================================
# 4) Small perturbation (optional)
# ============================================================

class SmallPerturbation:
    def __init__(
        self,
        noise_std=0.02,
        brightness=(0.95, 1.05),
        contrast=(0.95, 1.05),
        rotation=2.0,
        blur_prob=0.1,
    ):
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast
        self.rotation = rotation
        self.blur_prob = blur_prob

    def __call__(self, img: Image.Image):
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.normal(0, self.noise_std * 255, arr.shape)
        arr = np.clip(arr, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

        img = ImageEnhance.Brightness(img).enhance(
            np.random.uniform(*self.brightness)
        )
        img = ImageEnhance.Contrast(img).enhance(
            np.random.uniform(*self.contrast)
        )
        img = img.rotate(
            np.random.uniform(-self.rotation, self.rotation),
            fillcolor=(128, 128, 128)
        )

        if np.random.rand() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(0.5))

        return img


# ============================================================
# 5) Visualization
# ============================================================

def plot_btsc_samples(
    dataset: Dataset,
    output="btsc_samples.png",
    per_class=2
):
    by_class = defaultdict(list)
    for img, lbl in dataset:
        by_class[lbl].append(img)

    rows = len(by_class)
    cols = per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for r, (lbl, imgs) in enumerate(by_class.items()):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.set_title(f"Class {lbl}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def find_btsd_csv(csv_path: str) -> str:
    """
    Find BTSC/BTSD CSV file with robust path resolution.
    
    Args:
        csv_path: Path to CSV file (can be relative or absolute)
    Returns:
        Absolute path to CSV file
    Raises:
        FileNotFoundError if CSV file cannot be found
    """
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path
    
    # Try relative to current directory
    if os.path.exists(csv_path):
        return os.path.abspath(csv_path)
    
    # Try common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    common_paths = [
        csv_path,
        os.path.join(script_dir, csv_path),
        os.path.join(script_dir, "..", "..", "datasets", "BTSC", "labels.csv"),
        os.path.join(script_dir, "..", "..", "datasets", "BTSD", "labels.csv"),
        os.path.join(script_dir, "..", "..", "..", "datasets", "BTSC", "labels.csv"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(f"Could not find BTSC CSV file: {csv_path}. Tried: {common_paths}")


def find_btsd_images_dir(images_dir: str, csv_path: str) -> str:
    """
    Find BTSC/BTSD images directory with robust path resolution.
    
    Args:
        images_dir: Path to images directory (can be relative or absolute)
        csv_path: Path to CSV file (used to infer image location)
    Returns:
        Absolute path to images directory
    Raises:
        FileNotFoundError if images directory cannot be found
    """
    if os.path.isabs(images_dir) and os.path.isdir(images_dir):
        return images_dir
    
    # Try relative to current directory
    if os.path.isdir(images_dir):
        return os.path.abspath(images_dir)
    
    # Try common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.dirname(csv_path) if csv_path else script_dir
    
    common_paths = [
        images_dir,
        os.path.join(script_dir, images_dir),
        os.path.join(csv_dir, "Testing"),
        os.path.join(csv_dir, "Training"),
        os.path.join(os.path.dirname(csv_dir), "Testing"),
        os.path.join(script_dir, "..", "..", "datasets", "BTSC", "Testing"),
        os.path.join(script_dir, "..", "..", "datasets", "BTSD", "images"),
    ]
    
    for path in common_paths:
        if os.path.isdir(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(f"Could not find BTSC images directory: {images_dir}. Tried: {common_paths}")


def plot_selected_images_with_labels(
    csv_file: str,
    images_root: str,
    output_file: str = "btsc_selected_images.png",
    images_per_class: int = 2
):
    """
    Plot selected BTSC images with semantic labels.
    
    Args:
        csv_file: Path to BTSC labels CSV
        images_root: Root directory containing BTSC images
        output_file: Output path for the plot
        images_per_class: Number of images to show per class
    """
    import random
    
    # Load annotations
    annotations = load_btsc_csv(csv_file)
    
    # Group by BTSC class
    by_btsc_class = defaultdict(list)
    for rel_path, btsc_id in annotations:
        if btsc_id in BTSC_TO_GTSRB:
            full_path = os.path.join(images_root, rel_path)
            if os.path.exists(full_path):
                by_btsc_class[btsc_id].append((full_path, BTSC_TO_GTSRB[btsc_id]))
    
    # Select images per class
    selected = {}
    for btsc_id, samples in by_btsc_class.items():
        random.shuffle(samples)
        selected[btsc_id] = samples[:images_per_class]
    
    # Create plot
    num_classes = len(selected)
    fig, axes = plt.subplots(num_classes, images_per_class, 
                             figsize=(images_per_class * 3, num_classes * 3))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    if images_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for row, (btsc_id, samples) in enumerate(sorted(selected.items())):
        gtsrb_label = BTSC_TO_GTSRB[btsc_id]
        semantic = BTSC_SEMANTIC.get(btsc_id, f"Class {btsc_id}")
        
        for col, (img_path, _) in enumerate(samples):
            ax = axes[row, col]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(f"BTSC {btsc_id}\nGTSRB {gtsrb_label}\n{semantic}", 
                                fontsize=10, rotation=0, ha='right', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{str(e)}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")


def plot_btsc_gtsrb_comparison(
    csv_file: str,
    images_root: str,
    gtsrb_root: str,
    output_file: str = "btsc_gtsrb_comparison.png",
    images_per_class: int = 1,
    apply_perturbation: bool = False
):
    """
    Plot comparison of BTSC and GTSRB images.
    Layout: Two rows - first row is GTSRB images, second row is BTSC images.
    Each column represents the same class/content.
    
    Args:
        csv_file: Path to BTSC labels CSV
        images_root: Root directory containing BTSC images
        gtsrb_root: Root directory containing GTSRB images
        output_file: Output path for the plot
        images_per_class: Number of image pairs to show per class
        apply_perturbation: Whether to apply small perturbations (not used if False)
    """
    import random
    
    # Load BTSC annotations
    annotations = load_btsc_csv(csv_file)
    
    # Group BTSC by class
    by_btsc_class = defaultdict(list)
    for rel_path, btsc_id in annotations:
        if btsc_id in BTSC_TO_GTSRB:
            full_path = os.path.join(images_root, rel_path)
            if os.path.exists(full_path):
                by_btsc_class[btsc_id].append(full_path)
    
    # Select BTSC images per class
    selected_btsc = {}
    for btsc_id, paths in by_btsc_class.items():
        random.shuffle(paths)
        selected_btsc[btsc_id] = paths[:images_per_class]
    
    # Find corresponding GTSRB images and organize by class
    class_data = []
    for btsc_id, btsc_paths in sorted(selected_btsc.items()):
        gtsrb_label = BTSC_TO_GTSRB[btsc_id]
        semantic = BTSC_SEMANTIC.get(btsc_id, f"Class {btsc_id}")
        
        # Find GTSRB images for this class
        gtsrb_class_dir = os.path.join(gtsrb_root, f"{gtsrb_label:05d}")
        if not os.path.isdir(gtsrb_class_dir):
            # Try alternative paths
            alt_paths = [
                os.path.join(gtsrb_root, "gtsrb", "GTSRB", "Training", f"{gtsrb_label:05d}"),
                os.path.join(gtsrb_root, "Training", f"{gtsrb_label:05d}"),
            ]
            for alt_path in alt_paths:
                if os.path.isdir(alt_path):
                    gtsrb_class_dir = alt_path
                    break
        
        gtsrb_images = []
        if os.path.isdir(gtsrb_class_dir):
            for fname in os.listdir(gtsrb_class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.ppm')):
                    gtsrb_images.append(os.path.join(gtsrb_class_dir, fname))
        
        random.shuffle(gtsrb_images)
        
        # For each BTSC image, find a corresponding GTSRB image
        for idx, btsc_path in enumerate(btsc_paths):
            gtsrb_path = gtsrb_images[idx] if idx < len(gtsrb_images) else (gtsrb_images[0] if gtsrb_images else None)
            class_data.append({
                'btsc_id': btsc_id,
                'gtsrb_label': gtsrb_label,
                'semantic': semantic,
                'btsc_path': btsc_path,
                'gtsrb_path': gtsrb_path
            })
    
    # Create comparison plot: 2 rows (GTSRB top, BTSC bottom), N columns (one per class)
    num_cols = len(class_data)
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 2.5, 6))
    
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col, pair in enumerate(class_data):
        # Top row: GTSRB image
        ax_gtsrb = axes[0, col]
        if pair['gtsrb_path'] and os.path.exists(pair['gtsrb_path']):
            try:
                img_gtsrb = Image.open(pair['gtsrb_path']).convert("RGB")
                ax_gtsrb.imshow(img_gtsrb)
                title = f"GTSRB {pair['gtsrb_label']}"
                if col == 0:
                    title += f"\n{pair['semantic']}"
                ax_gtsrb.set_title(title, fontsize=9)
            except Exception as e:
                ax_gtsrb.text(0.5, 0.5, f"Error\n{str(e)}", ha='center', va='center', fontsize=8)
                ax_gtsrb.set_title(f"GTSRB {pair['gtsrb_label']}", fontsize=9)
        else:
            ax_gtsrb.text(0.5, 0.5, "Not found", ha='center', va='center', fontsize=8)
            ax_gtsrb.set_title(f"GTSRB {pair['gtsrb_label']}", fontsize=9)
        ax_gtsrb.set_xticks([])
        ax_gtsrb.set_yticks([])
        
        # Bottom row: BTSC image
        ax_btsc = axes[1, col]
        try:
            img_btsc = Image.open(pair['btsc_path']).convert("RGB")
            ax_btsc.imshow(img_btsc)
            ax_btsc.set_title(f"BTSC {pair['btsc_id']}", fontsize=9)
        except Exception as e:
            ax_btsc.text(0.5, 0.5, f"Error\n{str(e)}", ha='center', va='center', fontsize=8)
            ax_btsc.set_title(f"BTSC {pair['btsc_id']}", fontsize=9)
        ax_btsc.set_xticks([])
        ax_btsc.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {output_file}")


# ============================================================
# 6) Property generation functions
# ============================================================

def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img: The image tensor (C, H, W) with values in [0, 1]
        eps: The maximum accepted epsilon perturbation of each pixel.
    Returns:
        A (H*W*C) x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clamp((img - eps), 0, 1)
    bounds[..., 1] = torch.clamp((img + eps), 0, 1)
    return bounds.view(-1, 2)


def save_vnnlib(
    input_bounds: torch.Tensor, 
    label: int, 
    spec_path: str, 
    total_output_class: int = 43
):
    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class (original GTSRB label).
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes (43 for full GTSRB).
    """
    with open(spec_path, "w") as f:
        f.write(f"; BTSD property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]:.8f}))\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))")


def create_instances_csv(
    vnnlib_files: list,
    csv_path: str
):
    """
    Creates a CSV file containing relative paths to the .vnnlib files.

    Args:
        vnnlib_files:
            List of paths to vnnlib files (can be relative or absolute).
        csv_path:
            Path where the CSV file will be saved.
    """
    # Convert to relative paths
    rel_paths = []
    for vnnlib_file in vnnlib_files:
        if os.path.isabs(vnnlib_file):
            # Convert absolute path to relative path
            rel_paths.append(os.path.relpath(vnnlib_file))
        else:
            rel_paths.append(vnnlib_file)
    
    with open(csv_path, "w") as f:
        for prop_path in rel_paths:
            f.write(prop_path + "\n")
    
    print(f"✅ CSV saved to: {csv_path} ({len(rel_paths)} instances)")


def predict_consensus(img: torch.Tensor, sessions: List, expected_label: Optional[int] = None):
    """
    Return predicted class if ALL ONNX models agree.
    If expected_label is provided, also verify that the consensus matches it.
    Otherwise return None.
    
    Args:
        img: Image tensor (C, H, W) with values in [0, 1]
        sessions: List of ONNX inference sessions
        expected_label: Expected label to verify against (optional)
    Returns:
        Consensus label if all models agree, None otherwise
    """
    preds = []
    
    for sess in sessions:
        # Get expected input shape from model
        input_shape = sess.get_inputs()[0].shape
        # Input shape is typically (batch, height, width, channels) or (batch, channels, height, width)
        # Extract height and width (assuming square images)
        if len(input_shape) == 4:
            # Determine if channels-first or channels-last
            if input_shape[1] == 3 or input_shape[1] == 1:  # channels-first: (batch, C, H, W)
                model_h, model_w = input_shape[2], input_shape[3]
                channels_first = True
            else:  # channels-last: (batch, H, W, C)
                model_h, model_w = input_shape[1], input_shape[2]
                channels_first = False
        else:
            # Fallback: assume square image from current size
            model_h = model_w = img.shape[1]
            channels_first = False
        
        # Resize image to match model's expected input size
        img_resized = TF.resize(img, (model_h, model_w))
        
        # Convert to numpy
        img_np = img_resized.numpy()
        
        # Convert to format expected by model: (batch, H, W, C) or (batch, C, H, W)
        if channels_first:
            # Already in (C, H, W), add batch dimension
            img_np = img_np[np.newaxis, ...]  # (1, C, H, W)
        else:
            # Convert (C, H, W) -> (H, W, C) then add batch
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
            img_np = img_np[np.newaxis, ...]  # (1, H, W, C)
        
        # Scale to [0, 255] range (models expect this based on generate_properties.py)
        img_np = (img_np * 255.0).astype(np.float32)
        
        input_name = sess.get_inputs()[0].name
        y = sess.run(None, {input_name: img_np})[0]
        
        # Get GTSRB 43-class prediction
        pred_gtsrb = int(np.argmax(y))
        # Only accept predictions that are in our used classes
        if pred_gtsrb in USED_GTSRB:
            preds.append(pred_gtsrb)
        else:
            return None  # Prediction not in used classes
    
    if len(preds) == 0:
        return None
    
    if all(p == preds[0] for p in preds):
        consensus = preds[0]
        # If expected_label is provided, verify it matches
        if expected_label is not None and consensus != expected_label:
            return None
        return consensus
    return None


# ============================================================
# 7) Main property generation
# ============================================================

def generate_properties(
    btsd_root: str = "BTSD",
    btsd_csv: str = "BTSD/labels.csv",
    gtsrb_root: str = "GTSRB/Train",
    output_dir: str = "./btsd",
    onnx_models: Optional[List[str]] = None,
    epsilons: List[float] = [1/255, 2/255, 3/255, 4/255],
    samples_per_class: int = 10,
    img_size: int = 32,
    seed: int = 42,
):
    """
    Generate VNNLIB properties for BTSD dataset.
    
    Args:
        btsd_root: Root directory for BTSD images
        btsd_csv: Path to BTSD labels CSV file
        gtsrb_root: Root directory for GTSRB training images
        output_dir: Output directory for generated properties
        onnx_models: List of paths to ONNX models (optional)
        epsilons: List of epsilon values for perturbation
        samples_per_class: Number of samples per class to generate
        img_size: Target image size (height/width)
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("BTSD Property Generator")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Load ONNX models if provided
    sessions = []
    if onnx_models:
        for model_path in onnx_models:
            if not os.path.exists(model_path):
                print(f"Warning: ONNX model not found: {model_path}")
                continue
            sessions.append(onnxrun.InferenceSession(model_path))
        print(f"Loaded {len(sessions)} ONNX models")
    
    # Load datasets
    print("\nLoading datasets...")
    ann = load_btsc_csv(btsd_csv)
    btsd_dataset = OriginalLabels(
        BTSCMappedDataset(btsd_root, ann, transform)
    )
    
    gtsrb_dataset = OriginalLabels(
        GTSRBSubset(gtsrb_root, USED_GTSRB, transform)
    )
    
    print(f"BTSD samples: {len(btsd_dataset)}")
    print(f"GTSRB samples: {len(gtsrb_dataset)}")
    print(f"Classes: {len(USED_GTSRB)} (GTSRB labels: {sorted(set(BTSC_TO_GTSRB.values()))})")
    
    # Select samples per class from BTSD
    np.random.seed(seed)
    selected_samples = defaultdict(list)
    selected_labels = defaultdict(list)
    
    # Group BTSD samples by original GTSRB class
    by_class = defaultdict(list)
    for idx in range(len(btsd_dataset)):
        img, label = btsd_dataset[idx]
        by_class[label].append((img, label))
    
    # Get unique GTSRB classes used
    used_gtsrb_classes = sorted(set(BTSC_TO_GTSRB.values()))
    
    print("\nSelecting samples...")
    for gtsrb_class in used_gtsrb_classes:
        class_samples = by_class[gtsrb_class]
        np.random.shuffle(class_samples)
        
        count = 0
        for img, label in class_samples:
            if count >= samples_per_class:
                break
            
            # Verify with ONNX models if available
            if sessions:
                consensus = predict_consensus(img, sessions, expected_label=label)
                if consensus is None:
                    continue
            
            selected_samples[gtsrb_class].append(img)
            selected_labels[gtsrb_class].append(label)
            count += 1
        
        print(f"Class {gtsrb_class}: selected {len(selected_samples[gtsrb_class])} samples")
    
    total_samples = sum(len(samples) for samples in selected_samples.values())
    print(f"\nTotal selected samples: {total_samples}")
    
    # Generate properties for each epsilon
    # Use full GTSRB class space (43 classes)
    total_output_classes = 43
    
    for eps in epsilons:
        eps_num = int(eps * 255)
        eps_str = f"{eps_num}over255"
        vnnlib_files = []
        idx = 0
        
        print(f"\n{'='*60}")
        print(f"Generating properties for epsilon = {eps:.6f} ({eps_num}/255)")
        print(f"{'='*60}")
        
        # Iterate over GTSRB classes in sorted order
        for gtsrb_class in sorted(selected_samples.keys()):
            for sample_idx, img in enumerate(selected_samples[gtsrb_class]):
                bounds = create_input_bounds(img, eps)
                
                # Generate filename: btsd_{gtsrb_class}_{sample_idx}_{eps_value}.vnnlib
                spec_name = f"btsd_{gtsrb_class}_{sample_idx}_{eps_str}.vnnlib"
                spec_path = os.path.join(output_dir, spec_name)
                
                # Save vnnlib file with original GTSRB label
                save_vnnlib(bounds, gtsrb_class, spec_path, total_output_classes)
                vnnlib_files.append(spec_name)
                idx += 1
                
                if idx % 10 == 0:
                    print(f"  Generated {idx}/{total_samples} properties...")
        
        # Create CSV file for this epsilon
        csv_filename = os.path.join(output_dir, f"btsd_instances_eps{eps_str}.csv")
        create_instances_csv(vnnlib_files, csv_filename)
    
    print(f"\n{'='*60}")
    print("✅ Property generation complete!")
    print(f"{'='*60}")
    print(f"Generated properties for {total_samples} samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in epsilons]}")
    print(f"\nCSV files created:")
    for eps in epsilons:
        eps_num = int(eps * 255)
        print(f"  - {output_dir}/btsd_instances_eps{eps_num}over255.csv")


# ============================================================
# 8) Example usage
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate VNNLIB properties for BTSD dataset')
    parser.add_argument('--btsd_root', type=str, default='BTSD/images',
                        help='Root directory for BTSD images')
    parser.add_argument('--btsd_csv', type=str, default='BTSD/labels.csv',
                        help='Path to BTSD labels CSV file')
    parser.add_argument('--gtsrb_root', type=str, default='GTSRB/Train',
                        help='Root directory for GTSRB training images')
    parser.add_argument('--output_dir', type=str, default='./btsd_properties',
                        help='Output directory for generated properties')
    parser.add_argument('--onnx_models', type=str, nargs='+', default=None,
                        help='Paths to ONNX models (optional)')
    parser.add_argument('--epsilons', type=str, nargs='+',
                        default=['1/255', '2/255', '3/255', '4/255'],
                        help='Epsilon values (e.g., 1/255 2/255 or 0.00392)')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Number of samples per class')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Target image size (height/width)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Parse epsilon strings (supports fractions like "1/255" and decimals like "0.004")
    def parse_eps(s):
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    args.epsilons = [parse_eps(e) for e in args.epsilons]

    # Default ONNX models if not specified
    if args.onnx_models is None:
        default_models = [
            './3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx',
            './3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx',
            './3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx'
        ]
        # Only use models that exist
        args.onnx_models = [m for m in default_models if os.path.exists(m)]
        if args.onnx_models:
            print(f"Using default ONNX models: {args.onnx_models}")
        else:
            print("Warning: No ONNX models found, proceeding without verification")
            args.onnx_models = None
    
    generate_properties(
        btsd_root=args.btsd_root,
        btsd_csv=args.btsd_csv,
        gtsrb_root=args.gtsrb_root,
        output_dir=args.output_dir,
        onnx_models=args.onnx_models,
        epsilons=args.epsilons,
        samples_per_class=args.samples_per_class,
        img_size=args.img_size,
        seed=args.seed,
    )


# python generate_btsd_properties.py \
#   --btsd_root /data/judy/code/veri-generalization/datasets/BTSC/Testing \
#   --btsd_csv /data/judy/code/veri-generalization/datasets/BTSC/labels.csv \
#   --gtsrb_root /data/judy/code/veri-generalization/datasets/GTSRB/Train \
#   --output_dir /data/judy/code/veri-generalization/generate_properties/gtsrb/btsd \
#   --onnx_models \
#     /data/judy/code/veri-generalization/generate_properties/gtsrb/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx \
#     /data/judy/code/veri-generalization/generate_properties/gtsrb/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx \
#     /data/judy/code/veri-generalization/generate_properties/gtsrb/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx \
#   --samples_per_class 



# cd generate_properties/gtsrb && /data/judy/conda/envs/alpha-beta-crown/bin/python generate_btsd_properties.py \
#   --btsd_root /path/to/datasets/BTSC/Testing \
#   --btsd_csv /path/to/datasets/BTSC/labels.csv \
#   --gtsrb_root /path/to/datasets/GTSRB/gtsrb/GTSRB/Training \
#   --output_dir ./btsd \
#   --onnx_models 3_30_30_*.onnx 3_48_48_*.onnx 3_64_64_*.onnx \
#   --samples_per_class 10
# Note: --gtsrb_root must point to GTSRB Training (per-class dirs 00000/, 00001/, ...), not Final_Test/Images.