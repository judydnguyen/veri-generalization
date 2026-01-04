#!/usr/bin/env python3
"""
Generate MNIST properties with 10 samples per class for different epsilon values.

This script:
- Loads MNIST test dataset
- Selects 10 samples per class (100 samples total)
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


def load_data_per_class(
    data_dir: str = "./tmp",
    samples_per_class: int = 10,
    random: bool = True,
    onnx_models: list = None
) -> tuple:
    """
    Loads MNIST data with exactly samples_per_class samples for each class.

    Args:
        data_dir:
            The directory to store the full MNIST dataset.
        samples_per_class:
            The number of samples to extract per class.
        random:
            If true, random image indices are used, otherwise the first images
            are used.
        onnx_models:
            List of paths to ONNX model files for verification. If None, skips verification.
    Returns:
        A tuple of lists (images, labels).
    """
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(mnist_test, batch_size=10000,
                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(mnist_test, batch_size=10000)

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
            else:
                print(f"Warning: ONNX model {model_path} not found. Skipping verification.")
    
    # Select samples_per_class samples for each class
    for class_label in range(10):  # MNIST has 10 classes
        class_samples = 0
        indices = class_indices[class_label]
        
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
    
    return selected_images, selected_labels


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
        f.write(f"; Mnist property with label: {label}.\n")

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


def main():
    """Main function to generate properties."""
    
    # Configuration
    samples_per_class = 10
    epsilons = [1/255, 2/255, 3/255, 4/255]
    data_dir = "../../datasets/MNIST"
    
    # Output directory for all generated files
    output_dir = "./mnist"
    
    # Optional: ONNX models for verification (can be None)
    onnx_models = [
        "./mnist-net_256x2.onnx",
        "./mnist-net_256x4.onnx",
        "./mnist-net_256x6.onnx"
    ]
    
    print("="*60)
    print("MNIST Property Generator")
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
    print(f"\nLoading MNIST data with {samples_per_class} samples per class...")
    images, labels = load_data_per_class(
        data_dir=data_dir,
        samples_per_class=samples_per_class,
        random=True,
        onnx_models=onnx_models
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
            
            # Save vnnlib file
            save_vnnlib(input_bounds, int(label), spec_path)
            vnnlib_files.append(f"prop_{i}_{eps_str}.vnnlib")
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(images)} properties...")
        
        # Create CSV file for this epsilon
        eps_num = int(eps * 255)
        csv_filename = os.path.join(output_dir, f"mnistfc_instances_eps{eps_num}over255.csv")
        create_instances_csv(vnnlib_files, csv_filename)
    
    print(f"\n{'='*60}")
    print("✅ Property generation complete!")
    print(f"{'='*60}")
    print(f"Generated properties for {len(images)} samples")
    print(f"Epsilon values: {[f'{int(eps*255)}/255' for eps in epsilons]}")
    print(f"\nCSV files created:")
    for eps in epsilons:
        eps_num = int(eps * 255)
        print(f"  - {output_dir}/mnistfc_instances_eps{eps_num}over255.csv")


if __name__ == '__main__':
    main()

# EXAMPLE USAGE: python generate_mnist_properties.py 12345
