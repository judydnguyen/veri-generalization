#!/usr/bin/env python3
"""
Generate GTSRB properties with 5 samples per class for different epsilon values.

This script:
- Loads GTSRB test dataset
- Selects 5 samples per class (215 samples total for 43 classes)
- Generates vnnlib property files for each sample with different epsilon values
- Creates a CSV file for each epsilon value listing all vnnlib instances
"""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict


def load_data_per_class(
    data_dir: str = "../../datasets/GTSRB",
    samples_per_class: int = 5,
    random: bool = True,
    num_classes: int = 43
) -> tuple:
    """
    Loads GTSRB data with exactly samples_per_class samples for each class.

    Args:
        data_dir:
            The directory to store the full GTSRB dataset.
        samples_per_class:
            The number of samples to extract per class.
        random:
            If true, random image indices are used, otherwise the first images
            are used.
        num_classes:
            Total number of classes in GTSRB (43).
    Returns:
        A tuple of lists (images, labels).
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    gtsrb_test = torchvision.datasets.GTSRB(data_dir, split="test", download=True, transform=transform)

    # Load all test data
    loader_test = DataLoader(gtsrb_test, batch_size=len(gtsrb_test), shuffle=False)
    images, labels = next(iter(loader_test))

    # Group images by class
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[int(label)].append(i)

    selected_images, selected_labels = [], []

    # Select samples_per_class samples for each class
    for class_label in range(num_classes):
        class_samples = 0
        indices = class_indices[class_label]

        if random:
            np.random.shuffle(indices)

        for idx in indices:
            if class_samples >= samples_per_class:
                break

            selected_images.append(images[idx])
            selected_labels.append(labels[idx])
            class_samples += 1

        if class_samples < samples_per_class:
            print(f"Warning: Only found {class_samples} samples for class {class_label}")

    print(f"Selected {len(selected_images)} samples total")
    class_counts = [0] * num_classes
    for l in selected_labels:
        class_counts[int(l)] += 1
    print(f"Class distribution: {class_counts}")

    return selected_images, selected_labels


def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Creates input bounds for the given image and epsilon.

    Args:
        img:
            The image tensor (C, H, W).
        eps:
           The maximum accepted epsilon perturbation of each pixel.
    Returns:
        A img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((img - eps), 0, 1)
    bounds[..., 1] = torch.clip((img + eps), 0, 1)
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
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes (43 for GTSRB).
    """
    with open(spec_path, "w") as f:
        f.write(f"; GTSRB property with label: {label}.\n")

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
            rel_paths.append(os.path.relpath(vnnlib_file))
        else:
            rel_paths.append(vnnlib_file)

    with open(csv_path, "w") as f:
        for prop_path in rel_paths:
            f.write(prop_path + "\n")

    print(f"CSV saved to: {csv_path} ({len(rel_paths)} instances)")


def main():
    """Main function to generate properties."""

    # Configuration
    samples_per_class = 5
    num_classes = 43
    epsilons = [1/255, 2/255, 3/255, 4/255]
    data_dir = "../../datasets/GTSRB"

    # Output directory for all generated files
    output_dir = "./gtsrb"

    print("="*60)
    print("GTSRB Property Generator")
    print("="*60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Remove all existing .vnnlib files in output directory
    print("\nCleaning up existing .vnnlib files...")
    vnnlib_files = [f for f in os.listdir(output_dir) if f.endswith('.vnnlib')]
    if vnnlib_files:
        for f in vnnlib_files:
            os.remove(os.path.join(output_dir, f))
        print(f"  Removed {len(vnnlib_files)} existing .vnnlib files")
    else:
        print("  No existing .vnnlib files found")

    # Set random seed if provided
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"Using random seed: {seed}")
        except ValueError:
            print(f"Warning: Invalid seed '{sys.argv[1]}'. Using default random seed.")

    # Load data
    print(f"\nLoading GTSRB data with {samples_per_class} samples per class...")
    images, labels = load_data_per_class(
        data_dir=data_dir,
        samples_per_class=samples_per_class,
        random=True,
        num_classes=num_classes
    )

    # Generate properties for each epsilon
    for eps in epsilons:
        print(f"\n{'='*60}")
        print(f"Generating properties for epsilon = {eps:.6f} ({eps*255:.2f}/255)")
        print(f"{'='*60}")

        vnnlib_files = []

        for i, (image, label) in enumerate(zip(images, labels)):
            # Create input bounds
            input_bounds = create_input_bounds(image, eps)

            # Generate filename: prop_{index}_{eps_value}.vnnlib
            eps_num = int(eps * 255)
            eps_str = f"{eps_num}over255"
            spec_path = os.path.join(output_dir, f"prop_{i}_{eps_str}.vnnlib")

            # Save vnnlib file
            save_vnnlib(input_bounds, int(label), spec_path, num_classes)
            vnnlib_files.append(f"prop_{i}_{eps_str}.vnnlib")

            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{len(images)} properties...")

        # Create CSV file for this epsilon
        eps_num = int(eps * 255)
        csv_filename = os.path.join(output_dir, f"gtsrb_instances_eps{eps_num}over255.csv")
        create_instances_csv(vnnlib_files, csv_filename)

    print(f"\n{'='*60}")
    print("Property generation complete!")
    print(f"{'='*60}")
    print(f"Generated properties for {len(images)} samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in epsilons]}")
    print(f"\nCSV files created:")
    for eps in epsilons:
        eps_num = int(eps * 255)
        print(f"  - {output_dir}/gtsrb_instances_eps{eps_num}over255.csv")


if __name__ == '__main__':
    main()

# EXAMPLE USAGE: python generate_gtsrb_properties.py 42
