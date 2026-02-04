#!/usr/bin/env python3
"""
Generate VNNLIB properties for PixMix OOD GTSRB images.

Selection:
- Exactly 10 OOD samples per class (using original GTSRB labels)
- Selected only if ALL ONNX models agree on prediction
- Uses PixMix OOD images from gtsrb_pixmix_ood directory

Property (OOD):
- Standard robustness property: correct class must have highest logit
- Encoded as OR-of-ANDs (VNNLIB-safe, classifier-style)
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import onnxruntime as onnxrun
from collections import defaultdict
import csv
import re


# ============================================================
# Configuration
# ============================================================

OOD_ROOT = "/home/judy/code/veri-generalization/gtsrb_pixmix_ood"
OUTPUT_DIR = "./gtsrb_ood_pixmix"

EPSILONS = [1/255, 2/255, 3/255, 4/255]
SAMPLES_PER_CLASS = 10

# GTSRB ONNX models (different input sizes)
ONNX_MODELS = [
    "./3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx",
    "./3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx",
    "./3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx"
]

NUM_CLASSES = 43  # GTSRB has 43 classes
IMG_SIZE = 32  # Default image size for GTSRB


# ============================================================
# Helpers
# ============================================================

def load_image(path):
    """Load GTSRB-style RGB image."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(img)  # [3, H, W] with values in [0, 1]


def create_input_bounds(img: torch.Tensor, eps: float):
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


def predict_consensus(img: torch.Tensor, sessions, expected_label=None):
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
        preds.append(pred_gtsrb)
    
    if len(preds) == 0:
        return None
    
    if all(p == preds[0] for p in preds):
        consensus = preds[0]
        # If expected_label is provided, verify it matches
        if expected_label is not None and consensus != expected_label:
            return None
        return consensus
    return None


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
        f.write(f"; GTSRB PixMix OOD property with label: {label}.\n")

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


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("GTSRB PixMix OOD Property Generator")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ONNX models
    sessions = []
    for m in ONNX_MODELS:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Missing ONNX model: {m}")
        sessions.append(onnxrun.InferenceSession(m))

    print(f"Loaded {len(sessions)} ONNX models")

    # Select OOD samples per class
    selected = defaultdict(list)

    print("\nSelecting OOD samples...")
    for c in range(NUM_CLASSES):
        class_dir = os.path.join(OOD_ROOT, str(c))
        if not os.path.isdir(class_dir):
            print(f"Class {c}: directory not found, skipping")
            continue

        # Get all image files in the class directory
        image_files = sorted([f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))])
        
        for fname in image_files:
            if len(selected[c]) >= SAMPLES_PER_CLASS:
                break

            img_path = os.path.join(class_dir, fname)
            try:
                img = load_image(img_path)
                consensus = predict_consensus(img, sessions, expected_label=c)
                if consensus is not None:
                    selected[c].append(img)
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue

        print(f"Class {c}: selected {len(selected[c])} samples")

    # Report selection
    total = sum(len(selected[c]) for c in range(NUM_CLASSES))
    print(f"\nTotal selected OOD samples: {total}")

    if total == 0:
        print("❌ No samples selected! Check ONNX models and image paths.")
        return

    # Generate properties for each epsilon
    for eps in EPSILONS:
        eps_num = int(eps * 255)
        eps_str = f"{eps_num}over255"
        vnnlib_files = []
        idx = 0

        print(f"\n{'='*60}")
        print(f"Generating properties for epsilon = {eps:.6f} ({eps_num}/255)")
        print(f"{'='*60}")

        for c in range(NUM_CLASSES):
            for sample_idx, img in enumerate(selected[c]):
                bounds = create_input_bounds(img, eps)

                # Generate filename: ood_{class}_{sample_idx}_{eps_value}.vnnlib
                spec_name = f"ood_{c}_{sample_idx}_{eps_str}.vnnlib"
                spec_path = os.path.join(OUTPUT_DIR, spec_name)

                # Save vnnlib file
                save_vnnlib(bounds, c, spec_path, NUM_CLASSES)
                vnnlib_files.append(spec_name)
                idx += 1

                if idx % 10 == 0:
                    print(f"  Generated {idx}/{total} properties...")

        # Create CSV file for this epsilon
        csv_filename = os.path.join(OUTPUT_DIR, f"gtsrb_ood_instances_eps{eps_str}.csv")
        create_instances_csv(vnnlib_files, csv_filename)

    print(f"\n{'='*60}")
    print("✅ OOD property generation complete!")
    print(f"{'='*60}")
    print(f"Generated properties for {total} OOD samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in EPSILONS]}")
    print(f"\nCSV files created:")
    for eps in EPSILONS:
        eps_num = int(eps * 255)
        print(f"  - {OUTPUT_DIR}/gtsrb_ood_instances_eps{eps_num}over255.csv")


if __name__ == "__main__":
    main()
