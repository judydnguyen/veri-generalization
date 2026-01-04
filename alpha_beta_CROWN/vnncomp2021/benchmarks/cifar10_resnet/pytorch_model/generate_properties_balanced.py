############################################################
#    CIFAR10-ResNet benchmark (for VNN Comp 2021)          #
#    Balanced property generation (no PGD, no inference)   #
#                                                          #
# Copyright (C) 2021  Shiqi Wang (sw3215@columbia.edu)     #
# Copyright (C) 2021  Huan Zhang (huan@huan-zhang.com)     #
# Copyright (C) 2021  Kaidi Xu (xu.kaid@northeastern.edu)  #
#                                                          #
# This program is licenced under the BSD 2-Clause License  #
############################################################

import os
import argparse
import csv
import random

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

cifar10_mean = (0.4914, 0.4822, 0.4465)  # np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # np.std(train_set.train_data, axis=(0,1,2))/255


def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = False) -> tuple:
    """
    Loads the cifar10 data.

    Args:
        data_dir:
            The directory to store the full CIFAR10 dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))


def load_cifar10c_data(
    data_dir: str = None,
    num_imgs: int = 25,
    random: bool = False
) -> tuple:
    """
    Loads CIFAR10-C data from an image folder.

    Args:
        data_dir:
            The directory containing the CIFAR10-C images. This should be an image folder layout.
            If None, will raise an error (should be set by caller).
        num_imgs:
            The number of images to extract from the folder.
        random:
            If True, randomly sample num_imgs images from the folder.
            If False, take the first num_imgs images in dataset order.

    Returns:
        A tuple of tensors (images, labels), similar to load_data().
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified for CIFAR10-C dataset.")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir '{data_dir}' does not exist or is not a directory.")

    # Simple normalization to tensor, same as CIFAR10 loader
    trns_norm = trans.ToTensor()

    # ImageFolder will treat each subdirectory as a "class"
    cifar10c = dset.ImageFolder(root=data_dir, transform=trns_norm)

    if len(cifar10c) == 0:
        raise ValueError(f"No images found in directory '{data_dir}'.")

    num = min(num_imgs, len(cifar10c))

    if random:
        # Sample num random indices without replacement
        idxs = np.random.choice(len(cifar10c), size=num, replace=False)
        subset_samp = sampler.SubsetRandomSampler(idxs)
        loader = DataLoader(cifar10c, batch_size=num, sampler=subset_samp)
    else:
        # Just take the first `num` samples in order
        loader = DataLoader(cifar10c, batch_size=num, shuffle=False)

    images, labels = next(iter(loader))
    return images, labels


def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: tuple = (0.4914, 0.4822, 0.4465),
                        std: tuple = (0.2471, 0.2435, 0.2616)) -> torch.Tensor:
    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
        mean:
            The channel-wise means.
        std:
            The channel-wise standard deviation.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """
    mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.tensor(std, device=img.device).view(-1, 1, 1)

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32, device=img.device)
    bounds[..., 0] = (torch.clip((img - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((img + eps), 0, 1) - mean) / std

    return bounds.view(-1, 2)


def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):
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
    # Convert to CPU if on GPU for file I/O
    if input_bounds.is_cuda:
        input_bounds = input_bounds.cpu()
    
    with open(spec_path, "w") as f:
        f.write(f"; CIFAR10/CIFAR10-C property with label: {label}.\n")

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
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1].item():.8f}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0].item():.8f}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")


def create_vnnlib_balanced(args):
    """
    Balancedly selects samples from each class and creates vnnlib properties.
    No PGD attack, no model inference checks - just balanced sampling and property generation.

    Args:
        args: Command line arguments containing:
            - num_images: Total number of images to generate properties for
            - epsilons: Space-separated epsilon values (e.g., "2/255 4/255")
            - seed: Random seed for reproducibility
            - data_dir: Directory to store/load CIFAR10 data
            - output_dir: Directory to save vnnlib files
            - device: Device to use ('cpu' or 'gpu')
    """
    num_imgs = args.num_images
    epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
    num_classes = 10
    samples_per_class = num_imgs // num_classes
    
    dataset_type = getattr(args, 'dataset', 'cifar10').lower()
    
    print(f"===== Balanced property generation =====")
    print(f"Dataset: {dataset_type}")
    print(f"Total images: {num_imgs}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Epsilons: {epsilons}")
    print(f"Seed: {args.seed}")

    # Create output directory
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Properties will be saved in: {output_dir}")

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'gpu':
            torch.cuda.manual_seed_all(args.seed)

    # Load data based on dataset type
    data_dir = getattr(args, 'data_dir', None)
    
    if dataset_type == 'cifar10c':
        # Use CIFAR10-C dataset
        if data_dir is None:
            # Try to find CIFAR10-C in common locations relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                "/home/judy/code/unlearning-verification/alpha_beta_CROWN/complete_verifier/datasets/cifar10c_by_class",
                os.path.join(script_dir, "../../../complete_verifier/datasets/cifar10c_by_class"),
                os.path.join(script_dir, "../../../../complete_verifier/datasets/cifar10c_by_class"),
                "../complete_verifier/datasets/cifar10c_by_class",
                "../../complete_verifier/datasets/cifar10c_by_class",
                "./datasets/cifar10c_by_class",
            ]
            data_dir = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.isdir(abs_path):
                    data_dir = abs_path
                    break
            
            if data_dir is None:
                raise FileNotFoundError(
                    f"CIFAR10-C dataset not found. Please specify --data_dir. "
                    f"Tried: {possible_paths}"
                )
        print(f"Loading CIFAR10-C data from: {data_dir}")
        # Load a large number of images to ensure we have enough for balanced sampling
        # The function will load up to the available number
        max_images = 50000  # CIFAR10-C typically has many more images
        images, labels = load_cifar10c_data(data_dir=data_dir, num_imgs=max_images, random=True)
        print(f"Loaded {len(images)} images from CIFAR10-C")
    else:
        # Use standard CIFAR10 dataset
        if data_dir is None:
            data_dir = './tmp'
        print(f"Loading CIFAR10 data from: {data_dir}")
        images, labels = load_data(data_dir=data_dir, num_imgs=10000, random=True)
        print(f"Loaded {len(images)} images from CIFAR10")
    
    # Group indices by class
    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    # Select balanced samples per class
    selected_indices = []
    for c in range(num_classes):
        available = class_indices[c]
        if len(available) < samples_per_class:
            print(f"Warning: Class {c} has only {len(available)} samples, using all of them.")
            chosen = available
        else:
            chosen = random.sample(available, samples_per_class)
        selected_indices.extend(chosen)
        print(f"Class {c}: selected {len(chosen)} samples")

    print(f"Total selected samples: {len(selected_indices)}")

    # Move images to device if using GPU
    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')

    # Generate properties for each epsilon
    for eps in epsilons:
        cnt = 0
        eps_dir = os.path.join(output_dir, f"eps_{eps:.2f}")
        if not os.path.isdir(eps_dir):
            os.makedirs(eps_dir, exist_ok=True)

        for idx in selected_indices:
            image = images[idx].unsqueeze(0)  # Add batch dimension
            label = int(labels[idx])
            
            if device.type == 'cuda':
                image = image.cuda()

            # Create input bounds
            input_bounds = create_input_bounds(image, eps)
            
            # Save vnnlib file
            spec_path = os.path.join(eps_dir, f"prop_{cnt}_cls_{label}_eps_{eps:.2f}.vnnlib")
            save_vnnlib(input_bounds, label, spec_path)
            
            print(f"[eps={eps:.2f}] Sample #{cnt}: class {label}, idx {idx}")
            cnt += 1

        print(f"Generated {cnt} properties for epsilon {eps:.2f}")

    print("Balanced property generation complete!")


def create_csv_properties_only(output_dir: str, epsilons: list, csv_path: str):
    """
    Creates a CSV file listing all generated property paths.
    
    Args:
        output_dir: Directory where properties are saved (relative path from script location)
        epsilons: List of epsilon values
        csv_path: Path to save the CSV file
    """
    actual_properties = []
    for eps in epsilons:
        eps_dir = os.path.join(output_dir, f"eps_{eps:.2f}")
        if os.path.isdir(eps_dir):
            prop_files = sorted([f for f in os.listdir(eps_dir) if f.endswith('.vnnlib')])
            for prop_file in prop_files:
                # Use relative path from output_dir parent
                rel_path = os.path.join(os.path.basename(output_dir), f"eps_{eps:.2f}", prop_file)
                actual_properties.append([rel_path.replace('\\', '/')])
    
            final_csv_path_per_epsilon = os.path.join(eps_dir, f"sampled_specifications.csv")
            with open(final_csv_path_per_epsilon, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(actual_properties)
            print(f"Created CSV with {len(actual_properties)} properties at {final_csv_path_per_epsilon}")
        else:
            print(f"Warning: No property files found to list in CSV for epsilon {eps:.2f}")

    # if actual_properties:
    #     with open(csv_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerows(actual_properties)
    #     print(f"Created CSV with {len(actual_properties)} properties at {csv_path}")
    # else:
    #     print("Warning: No property files found to list in CSV")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate balanced vnnlib properties without PGD or inference checks')
    parser.add_argument('--num_images', type=int, default=50, help='Total number of images (will be balanced across classes)')
    parser.add_argument('--epsilons', type=str, default='2/255', help='Space-separated epsilon values (e.g., "2/255 4/255")')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar10c'], default='cifar10', help='Dataset to use: cifar10 or cifar10c')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/load data. For CIFAR10: default is ./tmp. For CIFAR10-C: default path is used if not specified.')
    parser.add_argument('--output_dir', type=str, default='../vnnlib_properties_balanced', help='Directory to save vnnlib files')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to use')
    parser.add_argument('--create_csv', action='store_true', help='Create a CSV file listing all properties')
    parser.add_argument('--csv_path', type=str, default='../cifar10_resnet_properties_balanced.csv', help='Path for CSV file')
    
    args = parser.parse_args()
    
    # Generate properties
    create_vnnlib_balanced(args)
    
    # Optionally create CSV
    if args.create_csv:
        epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
        create_csv_properties_only(args.output_dir, epsilons, args.csv_path)


############################################################
#    CIFAR10-ResNet benchmark (for VNN Comp 2021)          #
#    Balanced property generation (no PGD, no inference)   #
#                                                          #
# Copyright (C) 2021  Shiqi Wang (sw3215@columbia.edu)     #
# Copyright (C) 2021  Huan Zhang (huan@huan-zhang.com)     #
# Copyright (C) 2021  Kaidi Xu (xu.kaid@northeastern.edu)  #
#                                                          #
# This program is licenced under the BSD 2-Clause License  #
############################################################
# %%% Example usage:
# CIFAR10:
# python generate_properties_balanced.py --num_images 50 --epsilons "8/255" --seed 42 --device gpu --create_csv --csv_path ../cifar10_resnet_properties_balanced.csv --output_dir ../vnnlib_properties_balanced_eps_2_255_4_255
# python generate_properties_balanced.py --num_images 50 --epsilons "8/255" --seed 0 --device gpu --create_csv --csv_path ../cifar10_resnet_properties_balanced.csv
#
# CIFAR10-C:
# python generate_properties_balanced.py --dataset cifar10c --num_images 50 --epsilons "2/255 4/255 8/255 20/255" --seed 42 --device gpu --create_csv --output_dir ../vnnlib_properties_balanced_cifar10c
# python generate_properties_balanced.py --dataset cifar10c --data_dir /path/to/cifar10c_by_class --num_images 50 --epsilons "8/255" --seed 0 --device gpu
# %%%