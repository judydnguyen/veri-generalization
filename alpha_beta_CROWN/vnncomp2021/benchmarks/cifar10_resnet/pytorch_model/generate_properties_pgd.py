############################################################
#    CIFAR10-ResNet benchmark (for VNN Comp 2021)          #
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
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from resnet import resnet2b, resnet4b
from attack_pgd import attack_pgd

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

import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader, sampler


def load_cifar10c_data(
    data_dir: str = "/home/judy/code/unlearn_project/alpha_beta_CROWN/complete_verifier/datasets/cifar10c_by_class",
    num_imgs: int = 25,
    random: bool = False
) -> tuple:
    """
    Loads CIFAR10-C data from an image folder.

    Args:
        data_dir:
            The directory containing the CIFAR10-C images. This should be an image folder layout.
        num_imgs:
            The number of images to extract from the folder.
        random:
            If True, randomly sample num_imgs images from the folder.
            If False, take the first num_imgs images in dataset order.

    Returns:
        A tuple of tensors (images, labels), similar to load_data().
    """
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


# noinspection PyShadowingNames
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

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clip((img - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((img + eps), 0, 1) - mean) / std
    # print(bounds[..., 0].abs().sum(), bounds[..., 1].abs().sum())

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
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

    with open(spec_path, "w") as f:

        f.write(f"; CIFAR10 property with label: {label}.\n")

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
        # orignal separate version:
        # for i in range(total_output_class):
        #     if i != label:
        #         f.write(f"(assert (>= Y_{label} Y_{i}))\n")
        # f.write("\n")

        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")

def create_csv():
    name = ["model_name", "property_name", "timeout"]
    instance_list = []

    # 48 properties for resnet2b
    model_name = "resnet_2b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    assert os.path.exists("../vnnlib_properties_pgd_filtered/")
    for i in range(48):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_{i}_eps_0.008.vnnlib", "300"])

    # 24 properties for resnet2b
    model_name = "resnet_4b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    for i in range(24):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet4b_pgd_filtered/prop_{i}_eps_0.004.vnnlib", "300"])

    with open('../cifar10_resnet_instances.csv', 'w') as f:
        write = csv.writer(f)
        # write.writerow(fields)
        write.writerows(instance_list)

import os
import csv

def create_csv_properties_only(output_dir: str = None, epsilons: list = None, csv_path: str = None, model: str = None):
    """
    Creates a CSV file listing all generated property paths.
    
    Args:
        output_dir: Directory where properties are saved (relative path from script location)
        epsilons: List of epsilon values
        csv_path: Path to save the CSV file
        model: Model name (e.g., 'resnet2b' or 'resnet4b') - not used in new structure
    """
    if output_dir is None:
        output_dir = '../vnnlib_properties_pgd_filtered'
    if csv_path is None:
        csv_path = '../cifar10_resnet_properties_only.csv'
    
    actual_properties = []
    
    if epsilons is not None:
        # Scan for properties in epsilon-specific directories
        for eps in epsilons:
            eps_dir = os.path.join(output_dir, f"eps_{eps:.2f}")
            if os.path.isdir(eps_dir):
                prop_files = sorted([f for f in os.listdir(eps_dir) if f.endswith('.vnnlib')])
                for prop_file in prop_files:
                    rel_path = os.path.join(os.path.basename(output_dir), f"eps_{eps:.2f}", prop_file)
                    actual_properties.append([rel_path.replace('\\', '/')])
    else:
        # Fallback: scan all subdirectories for .vnnlib files
        if os.path.isdir(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.vnnlib'):
                        rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(output_dir))
                        actual_properties.append([rel_path.replace('\\', '/')])
    
    if actual_properties:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(actual_properties)
        print(f"Created CSV with {len(actual_properties)} properties at {csv_path}")
    else:
        print(f"Warning: No property files found to list in CSV for output_dir: {output_dir}")



def create_vnnlib(args):
    num_imgs = args.num_images
    dataset_type = getattr(args, 'dataset', 'cifar10').lower()
    print(f"===== model: {args.model} epsilons: {args.epsilons} total images: {args.num_images} =====")
    print(f"Dataset: {dataset_type}")
    print("deterministic", args.deterministic, "seed:", args.seed)
    epsilons = [eval(eps) for eps in args.epsilons.split(" ")]

    # Create output directory structure
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(args.model, "model_best.pth")
    print("loading model {} and properties saved in {}".format(model_path, output_dir))

    mu = torch.tensor(cifar10_mean).view(3,1,1)
    std = torch.tensor(cifar10_std).view(3,1,1)

    model = eval(args.model)()
    model.load_state_dict(torch.load(model_path, map_location='cpu')["state_dict"])
    if args.device == 'gpu':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.cuda()
        mu = mu.cuda()
        std = std.cuda()
        
    normalize = lambda X: (X - mu)/std

    if args.seed is not None:
        if args.device == 'gpu':
            torch.cuda.manual_seed_all(args.seed)
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load data based on dataset type
    data_dir = getattr(args, 'data_dir', None)
    if dataset_type == 'cifar10c':
        if data_dir is None:
            # Try to find CIFAR10-C in common locations
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
        images, labels = load_cifar10c_data(data_dir=data_dir, num_imgs=10000, random=not args.deterministic)
    else:
        if data_dir is None:
            data_dir = './tmp'
        print(f"Loading CIFAR10 data from: {data_dir}")
        images, labels = load_data(data_dir=data_dir, num_imgs=10000, random=not args.deterministic)

    for eps in epsilons:
        # Create epsilon-specific directory
        eps_dir = os.path.join(output_dir, f"eps_{eps:.2f}")
        if not os.path.isdir(eps_dir):
            os.makedirs(eps_dir, exist_ok=True)
        
        acc, pgd_acc = 0, 0
        cnt = 0
        for i in range(images.shape[0]):
            if cnt>=num_imgs:
                break

            # Load image and label.
            image, label = images[i], labels[i]
            image = image.unsqueeze(0)
            y = torch.tensor([label], dtype=torch.int64)
            if args.device == 'gpu':
                image = image.cuda()
                y = y.cuda()

            output = model(normalize(image))
            # Skip incorrect examples. 
            if output.max(1)[1] != label: 
                print("incorrect image {}".format(i))
                continue

            acc += 1
            # Skip attacked examples.
            perturbation = attack_pgd(model, X=image, y=y, epsilon=eps, alpha=eps / 2.0,
                    attack_iters=100, num_restarts=5, upper_limit=1.0, lower_limit=0.0, normalize=normalize)

            attack_image = image + perturbation
            assert (attack_image >= 0.).all()
            assert (attack_image <= 1.).all()
            assert perturbation.abs().max() <= eps
            attack_output = model(normalize((image + perturbation))).squeeze(0)
            attack_label = attack_output.argmax()

            if attack_label != label:
                print("pgd succeed image {}, label {}, against label {}".format(i, label, attack_label))
                continue

            pgd_acc += 1

            print("scanned images: {}, selected: {}, label {}".format(i, cnt, label))

            input_bounds = create_input_bounds(image, eps)
            spec_path = os.path.join(eps_dir, f"prop_{cnt}_eps_{eps:.2f}.vnnlib")
            save_vnnlib(input_bounds, label, spec_path)
            cnt += 1

        print(f"acc: {acc}, pgd_acc: {pgd_acc}, out of {i} samples for eps={eps:.2f}")


import torch
import numpy as np
import random
import os

def create_vnnlib_balanced(args):
    """
    Randomly select samples from each class (balanced),
    apply PGD attack, and create vnnlib specifications.
    """
    num_imgs = args.num_images
    dataset_type = getattr(args, 'dataset', 'cifar10').lower()
    epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
    samples_per_class = num_imgs // 10  # for CIFAR-10
    print(f"===== Balanced sampling per class: {samples_per_class} | Model: {args.model} =====")
    print(f"Dataset: {dataset_type}")

    # Create output directory structure
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    mu = torch.tensor(cifar10_mean).view(3, 1, 1)
    std = torch.tensor(cifar10_std).view(3, 1, 1)

    model_path = os.path.join(args.model, "model_best.pth")
    model = eval(args.model)()
    model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
    if args.device == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.cuda()
        mu, std = mu.cuda(), std.cuda()

    normalize = lambda X: (X - mu) / std

    # Random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "gpu":
            torch.cuda.manual_seed_all(args.seed)

    # Load data based on dataset type
    data_dir = getattr(args, 'data_dir', None)
    if dataset_type == 'cifar10c':
        if data_dir is None:
            # Try to find CIFAR10-C in common locations
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
        images, labels = load_cifar10c_data(data_dir=data_dir, num_imgs=10000, random=True)
    else:
        if data_dir is None:
            data_dir = './tmp'
        print(f"Loading CIFAR10 data from: {data_dir}")
        images, labels = load_data(data_dir=data_dir, num_imgs=10000, random=True)

    # Group indices by class
    class_indices = {c: [] for c in range(10)}
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    # Select random samples per class
    selected_indices = []
    for c in range(10):
        chosen = random.sample(class_indices[c], min(samples_per_class, len(class_indices[c])))
        selected_indices.extend(chosen)

    print(f"Selected {len(selected_indices)} total samples across all classes.")

    for eps in epsilons:
        # Create epsilon-specific directory
        eps_dir = os.path.join(output_dir, f"eps_{eps:.2f}")
        if not os.path.isdir(eps_dir):
            os.makedirs(eps_dir, exist_ok=True)
        
        cnt = 0
        for i in selected_indices:
            image, label = images[i], labels[i]
            image = image.unsqueeze(0)
            y = torch.tensor([label], dtype=torch.int64)
            if args.device == "gpu":
                image, y = image.cuda(), y.cuda()

            # Generate PGD perturbation regardless of model correctness
            perturbation = attack_pgd(
                model,
                X=image,
                y=y,
                epsilon=eps,
                alpha=eps / 2.0,
                attack_iters=100,
                num_restarts=5,
                upper_limit=1.0,
                lower_limit=0.0,
                normalize=normalize,
            )

            attack_image = torch.clamp(image + perturbation, 0, 1)
            assert perturbation.abs().max() <= eps

            print(f"[eps={eps:.2f}] class {label}, idx {i}, sample #{cnt}")

            input_bounds = create_input_bounds(image, eps)
            spec_path = os.path.join(eps_dir, f"prop_{cnt}_cls_{label}_eps_{eps:.2f}.vnnlib")
            save_vnnlib(input_bounds, label, spec_path)
            cnt += 1

        print(f"Generated {cnt} properties for epsilon {eps:.2f}")

    print("Balanced sample generation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vnnlib properties with PGD attack filtering')
    parser.add_argument('--model', type=str, default="resnet2b", choices=["resnet2b", "resnet4b"], help='Model to use')
    parser.add_argument('--num_images', type=int, default=50, help='Total number of images to generate properties for')
    parser.add_argument('--epsilons', type=str, default='2/255', help='Space-separated epsilon values (e.g., "2/255 4/255")')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Do not generate random examples; use dataset order instead.')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to use')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar10c'], default='cifar10', help='Dataset to use: cifar10 or cifar10c')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/load data. For CIFAR10: default is ./tmp. For CIFAR10-C: default path is used if not specified.')
    parser.add_argument('--output_dir', type=str, default='../vnnlib_properties_pgd_filtered', help='Directory to save vnnlib files')
    parser.add_argument('--create_csv', action='store_true', help='Create a CSV file listing all properties')
    parser.add_argument('--csv_path', type=str, default='../cifar10_resnet_properties_pgd_filtered.csv', help='Path for CSV file')
    parser.add_argument('--use_balanced', action='store_true', help='Use balanced sampling function instead of PGD filtering')
    
    args = parser.parse_args()

    if args.use_balanced:
        create_vnnlib_balanced(args)
        if args.create_csv:
            epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
            create_csv_properties_only(args.output_dir, epsilons, args.csv_path)
    else:
        create_vnnlib(args)
        if args.create_csv:
            epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
            create_csv_properties_only(args.output_dir, epsilons, args.csv_path)


# python generate_properties_pgd.py --model resnet4b --num_images 20 --epsilons "2/255 4/255 8/255" --dataset cifar10 --device gpu --create_csv