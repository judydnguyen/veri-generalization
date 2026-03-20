#!/usr/bin/env python3
"""
Generate VNNLIB properties for PixMix OOD MNIST images.

Selection:
- Exactly 10 OOD samples per class
- Selected only if ALL ONNX models agree on prediction

Property (OOD):
- There exists a maximal logit Y_k such that Y_k <= tau
- Encoded as OR-of-ANDs (VNNLIB-safe, classifier-style)
"""

import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as onnxrun
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

SEED = 42

OOD_ROOT = "../../mnist_pixmix_ood"  # Update this path to point to your mnist_pixmix_ood directory
OUTPUT_DIR = "./mnist_pixmix_ood"

EPSILONS = [1/255, 2/255, 3/255, 4/255]
TAU = 0.8
SAMPLES_PER_CLASS = 10
NUM_CLASSES = 10

ONNX_MODELS = [
    "./mnist-net_256x2.onnx",
    "./mnist-net_256x4.onnx",
    "./mnist-net_256x6.onnx"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_image(path):
    """Load MNIST-style grayscale image."""
    img = Image.open(path).convert("L")
    return transforms.ToTensor()(img)  # [1,28,28]


def create_input_bounds(img: torch.Tensor, eps: float):
    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((img - eps), 0, 1)
    bounds[..., 1] = torch.clip((img + eps), 0, 1)
    return bounds.view(-1, 2)


def predict_consensus(img, sessions, expected_label=None):
    """
    Return predicted class if ALL ONNX models agree.
    If expected_label is provided, also verify that the consensus matches it.
    Otherwise return None.
    """
    x = img.numpy().reshape(1, 784, 1)
    preds = []

    for sess in sessions:
        input_name = sess.get_inputs()[0].name
        y = sess.run(None, {input_name: x})[0]
        preds.append(int(np.argmax(y)))

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
    total_output_class: int = 10
):
    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """
    with open(spec_path, "w") as f:
        f.write(f"; Mnist - PixMix OOD property with label: {label}.\n")

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
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
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
    print("MNIST PixMix OOD Property Generator (Final)")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load ONNX models
    sessions = []
    for m in ONNX_MODELS:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Missing ONNX model: {m}")
        sessions.append(onnxrun.InferenceSession(m))

    print(f"Loaded {len(sessions)} ONNX models")

    # Select OOD samples per class
    selected = defaultdict(list)

    for c in range(NUM_CLASSES):
        class_dir = os.path.join(OOD_ROOT, str(c))
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if len(selected[c]) >= SAMPLES_PER_CLASS:
                break

            img_path = os.path.join(class_dir, fname)
            img = load_image(img_path)

            consensus = predict_consensus(img, sessions, expected_label=c)
            if consensus is not None:
                selected[c].append(img)

    # Report selection
    total = 0
    for c in range(NUM_CLASSES):
        print(f"Class {c}: selected {len(selected[c])} samples")
        total += len(selected[c])

    if total != NUM_CLASSES * SAMPLES_PER_CLASS:
        print("⚠️ Warning: Not all classes reached target sample count")

    print(f"Total selected OOD samples: {total}")

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
            for img in selected[c]:
                bounds = create_input_bounds(img, eps)

                # Generate filename: ood_{index}_{eps_value}.vnnlib
                # Format epsilon as fraction (e.g., "1over255" for 1/255)
                spec_name = f"ood_{idx}_{eps_str}.vnnlib"
                spec_path = os.path.join(OUTPUT_DIR, spec_name)

                # Save vnnlib file
                save_vnnlib(bounds, c, spec_path, NUM_CLASSES)
                vnnlib_files.append(spec_name)
                idx += 1

                if idx % 10 == 0:
                    print(f"  Generated {idx}/{total} properties...")

        # Create CSV file for this epsilon
        csv_filename = os.path.join(OUTPUT_DIR, f"mnist_ood_instances_eps{eps_str}.csv")
        create_instances_csv(vnnlib_files, csv_filename)

    print(f"\n{'='*60}")
    print("✅ OOD property generation complete!")
    print(f"{'='*60}")
    print(f"Generated properties for {total} OOD samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in EPSILONS]}")
    print(f"\nCSV files created:")
    for eps in EPSILONS:
        eps_num = int(eps * 255)
        print(f"  - {OUTPUT_DIR}/mnist_ood_instances_eps{eps_num}over255.csv")


if __name__ == "__main__":
    main()
