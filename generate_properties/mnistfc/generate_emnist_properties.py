#!/usr/bin/env python3
"""
Generate EMNIST properties with configurable samples per class for different epsilon values.

This script:
- Loads EMNIST dataset (various splits: byclass, bymerge, balanced, letters, digits, mnist)
- Selects N samples per class
- Generates vnnlib property files for each sample with different epsilon values
- Creates a CSV file for each epsilon value listing all vnnlib instances
"""

import os
import sys
import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import numpy as np
import onnxruntime as onnxrun
from collections import defaultdict


def load_emnist_data_per_class(
    data_dir: str = "./tmp",
    samples_per_class: int = 10,
    random: bool = True,
    onnx_models: list = None,
    split: str = "byclass"
) -> tuple:
    """
    Loads EMNIST data with exactly samples_per_class samples for each class.

    Args:
        data_dir:
            The directory to store the full EMNIST dataset.
        samples_per_class:
            The number of samples to extract per class.
        random:
            If true, random image indices are used, otherwise the first images
            are used.
        onnx_models:
            List of paths to ONNX model files for verification. If None, skips verification.
        split:
            EMNIST split to use: 'byclass' (62 classes), 'bymerge' (47 classes),
            'balanced' (47 classes), 'letters' (26 classes), 'digits' (10 classes),
            'mnist' (10 classes).
    Returns:
        A tuple of (images, labels, num_classes).
    """
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    trns_norm = trans.Compose([
        trans.Lambda(lambda img: trans.functional.rotate(img, -90)),  # rotate to upright
        trans.Lambda(lambda img: trans.functional.hflip(img)),        # fix mirror
        trans.ToTensor(),
        # Optional: use MNIST normalization if you're comparing to MNIST models
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    
    # Load EMNIST dataset with specified split
    print(f"Loading EMNIST-{split} dataset...")
    emnist_test = dset.EMNIST(data_dir, split=split, train=False, download=True, transform=trns_norm)
    
    # Get number of classes for this split
    num_classes = len(emnist_test.classes)
    print(f"Dataset has {num_classes} classes")
    
    # Load all test data
    batch_size = min(len(emnist_test), 50000)
    if random:
        indices = torch.randperm(len(emnist_test))[:batch_size].tolist()
        loader_test = DataLoader(emnist_test, batch_size=batch_size,
                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    else:
        loader_test = DataLoader(emnist_test, batch_size=batch_size)

    images, labels = next(iter(loader_test))
    
    # Group images by class
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[int(label)].append(i)
    
    selected_images, selected_labels = [], []
    
    # Load ONNX models if provided
    sessions = []
    if onnx_models:
        for model_path in onnx_models:
            if os.path.exists(model_path):
                sessions.append(onnxrun.InferenceSession(model_path))
                print(f"Loaded ONNX model: {model_path}")
            else:
                print(f"Warning: ONNX model {model_path} not found. Skipping verification.")
    
    # Select samples_per_class samples for each class
    for class_label in range(num_classes):
        class_samples = 0
        indices = class_indices[class_label]
        
        if len(indices) == 0:
            print(f"Warning: No samples found for class {class_label}")
            continue
        
        if random:
            np.random.shuffle(indices)
        
        for idx in indices:
            if class_samples >= samples_per_class:
                break
            
            # Verify with ONNX models if available
            if sessions:
                correctly_classified = True
                for sess in sessions:
                    input_name = sess.get_inputs()[0].name
                    result = np.argmax(sess.run(
                        None, 
                        {input_name: images[idx].numpy().reshape(1, 784, 1)}
                    )[0])
                    
                    if result != labels[idx]:
                        correctly_classified = False
                        break
                
                if not correctly_classified:
                    continue
            
            selected_images.append(images[idx])
            selected_labels.append(labels[idx])
            class_samples += 1
        
        if class_samples < samples_per_class:
            print(f"Warning: Only found {class_samples} samples for class {class_label}")
    
    print(f"Selected {len(selected_images)} samples total")
    print(f"Class distribution: {np.bincount([int(l) for l in selected_labels])}")
    
    return selected_images, selected_labels, num_classes


def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:
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


def save_vnnlib(
    input_bounds: torch.Tensor, 
    label: int, 
    spec_path: str, 
    total_output_class: int,
    dataset_name: str = "EMNIST"
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
        dataset_name:
            Name of the dataset for the comment.
    """
    with open(spec_path, "w") as f:
        f.write(f"; {dataset_name} property with label: {label}.\n")

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
        # if os.path.isabs(vnnlib_file):
        #     # Convert absolute path to relative path
        #     rel_paths.append(os.path.relpath(vnnlib_file))
        # else:
        rel_paths.append(vnnlib_file)
    
    with open(csv_path, "w") as f:
        for prop_path in rel_paths:
            f.write(prop_path + "\n")
    
    print(f"✅ CSV saved to: {csv_path} ({len(rel_paths)} instances)")


def main():
    """Main function to generate properties."""
    
    # Configuration
    samples_per_class = 10
    epsilons = [1/255, 2/255, 3/255, 4/255]
    data_dir = "../../datasets/EMNIST"
    
    # Output directory for all generated files
    output_dir = "./emnist"
    
    # EMNIST split selection
    # Options: 'byclass' (62), 'bymerge' (47), 'balanced' (47), 'letters' (26), 'digits' (10), 'mnist' (10)
    split = "digits"
    
    # Optional: ONNX models for verification (can be None)
    onnx_models = [
        "./emnist-net_256x2.onnx",
        "./emnist-net_256x4.onnx",
        "./emnist-net_256x6.onnx"
    ]
    
    print("="*60)
    print(f"EMNIST Property Generator - Split: {split}")
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
    
    # Check if ONNX models exist
    existing_models = [m for m in onnx_models if os.path.exists(m)]
    if not existing_models:
        print("\nWarning: No ONNX models found. Proceeding without verification.")
        onnx_models = None
    else:
        onnx_models = existing_models
        print(f"\nUsing {len(onnx_models)} ONNX models for verification")
    
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
    print(f"\nLoading EMNIST-{split} data with {samples_per_class} samples per class...")
    images, labels, num_classes = load_emnist_data_per_class(
        data_dir=data_dir,
        samples_per_class=samples_per_class,
        random=True,
        onnx_models=onnx_models,
        split=split
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
            # Format epsilon as fraction (e.g., "1over255" for 1/255)
            eps_num = int(eps * 255)
            eps_str = f"{eps_num}over255"
            spec_path = os.path.join(output_dir, f"prop_{i}_{eps_str}.vnnlib")
            # spec_path = f"prop_{i}_{eps_str}.vnnlib"
            
            # Save vnnlib file
            save_vnnlib(input_bounds, int(label), spec_path, num_classes, f"EMNIST-{split}")
            vnnlib_files.append(f"prop_{i}_{eps_str}.vnnlib")
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(images)} properties...")
        
        # Create CSV file for this epsilon
        eps_num = int(eps * 255)
        csv_filename = os.path.join(output_dir, f"emnistfc_instances_eps{eps_num}over255.csv")
        create_instances_csv(vnnlib_files, csv_filename)
    
    print(f"\n{'='*60}")
    print("✅ Property generation complete!")
    print(f"{'='*60}")
    print(f"Dataset: EMNIST-{split} ({num_classes} classes)")
    print(f"Generated properties for {len(images)} samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in epsilons]}")
    print(f"\nCSV files created:")
    for eps in epsilons:
        eps_num = int(eps * 255)
        print(f"  - emnist_{split}_instances_eps{eps_num}over255.csv")
    print(f"\nTo use a different EMNIST split, modify the 'split' variable:")
    print(f"  - 'byclass': 62 classes (0-9, A-Z, a-z)")
    print(f"  - 'bymerge': 47 classes (merged similar characters)")
    print(f"  - 'balanced': 47 classes (balanced class distribution)")
    print(f"  - 'letters': 26 classes (A-Z)")
    print(f"  - 'digits': 10 classes (0-9)")
    print(f"  - 'mnist': 10 classes (MNIST-like)")


if __name__ == '__main__':
    main()

# EXAMPLE USAGE:
# python generate_emnist_properties.py 12345
#
# To change the split, edit the 'split' variable in main():
# - split = "byclass"   # 62 classes (default)
# - split = "letters"   # 26 classes (letters only)
# - split = "digits"    # 10 classes (digits only)
# - split = "balanced"  # 47 classes (balanced distribution)