import os
import sys
import csv
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from alpha_beta_CROWN.complete_verifier.model_defs import mnist_fc


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "./data"
PIXMIX_ROOT = "./mnist_pixmix_ood"  # Path to PixMix OOD images

ARCH = "mnist_fc"

# Use relative paths based on script location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "model_A": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc.pt"),
    "model_B": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.03_acc93.17.pt"),
    "model_C": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.01_acc93.60.pt"),
    "model_D": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.05_acc93.24.pt"),
}

BATCH_SIZE = 256
N_EXAMPLES = 1000  # Number of test examples to use


# =========================
# Model definition
# =========================

def build_model(arch):
    if arch == "mnist_fc":
        return mnist_fc()
    raise ValueError(f"Unsupported arch: {arch}")


def load_model(arch, ckpt_path):
    model = build_model(arch)
    sd = torch.load(ckpt_path, map_location="cpu")

    # Handle common checkpoint formats
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model.load_state_dict(sd)
    model.eval()
    return model.to(DEVICE)


# =========================
# Dataset loading
# =========================

def load_mnist_dataset(n_examples=N_EXAMPLES):
    """Load MNIST test dataset."""
    print(f"Loading MNIST test dataset (using {n_examples} examples)...")
    transform = trans.ToTensor()
    dataset = dset.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    xs, ys = [], []
    seen = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= n_examples:
            break
    
    x_test = torch.cat(xs)[:n_examples].to(DEVICE)
    y_test = torch.cat(ys)[:n_examples].to(DEVICE)
    print(f"Loaded {len(x_test)} MNIST test examples")
    return x_test, y_test


def load_emnist_dataset(n_examples=N_EXAMPLES):
    """Load EMNIST digits test dataset."""
    print(f"Loading EMNIST digits test dataset (using {n_examples} examples)...")
    
    # EMNIST requires rotation and flip to match MNIST orientation
    transform = trans.Compose([
        trans.Lambda(lambda img: trans.functional.rotate(img, -90)),  # rotate to upright
        trans.Lambda(lambda img: trans.functional.hflip(img)),        # fix mirror
        trans.ToTensor(),
    ])
    
    dataset = dset.EMNIST(DATA_ROOT, split="digits", train=False, download=True, transform=transform)
    
    # Load all test data
    loader = DataLoader(dataset, batch_size=min(len(dataset), 10000), shuffle=False)
    
    xs, ys = [], []
    seen = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= n_examples:
            break
    
    x_test = torch.cat(xs)[:n_examples].to(DEVICE)
    y_test = torch.cat(ys)[:n_examples].to(DEVICE)
    print(f"Loaded {len(x_test)} EMNIST test examples")
    return x_test, y_test


def load_pixmix_dataset(n_examples=N_EXAMPLES):
    """Load MNIST PixMix OOD dataset."""
    print(f"Loading MNIST PixMix OOD dataset (using {n_examples} examples)...")
    
    # Try to find PixMix directory
    possible_dirs = [
        PIXMIX_ROOT,
        os.path.join(_BASE_DIR, "mnist_pixmix_ood"),
        os.path.join(_BASE_DIR, "..", "mnist_pixmix_ood"),
    ]
    
    pixmix_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            pixmix_dir = dir_path
            break
    
    if pixmix_dir is None:
        raise FileNotFoundError(f"Could not find PixMix OOD directory. Tried: {possible_dirs}")
    
    transform = trans.ToTensor()
    images, labels = [], []
    num_classes = 10
    
    # Load images from each class directory
    for class_label in range(num_classes):
        class_dir = os.path.join(pixmix_dir, str(class_label))
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory {class_dir} does not exist")
            continue
        
        # Get all PNG files in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        image_files.sort()
        
        for img_file in image_files:
            if len(images) >= n_examples:
                break
            
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert("L")
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(torch.tensor(class_label))
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        if len(images) >= n_examples:
            break
    
    if len(images) == 0:
        raise ValueError(f"No images found in PixMix directory: {pixmix_dir}")
    
    # Limit to n_examples
    images = images[:n_examples]
    labels = labels[:n_examples]
    
    x_test = torch.stack(images).to(DEVICE)
    y_test = torch.stack(labels).to(DEVICE)
    print(f"Loaded {len(x_test)} PixMix test examples")
    return x_test, y_test


# =========================
# Accuracy computation
# =========================

@torch.no_grad()
def accuracy(model, x, y):
    """
    Compute accuracy. Input x should be in shape [B, 1, 28, 28] or [B, 784].
    Model expects [B, 784, 1] based on training code, so we reshape.
    """
    correct = 0
    total = 0
    for i in range(0, x.size(0), BATCH_SIZE):
        xb = x[i:i+BATCH_SIZE]
        yb = y[i:i+BATCH_SIZE]
        # Reshape to [B, 784, 1] as expected by the model
        if xb.dim() == 4:  # [B, 1, 28, 28]
            xb = xb.view(xb.size(0), -1, 1)
        elif xb.dim() == 2:  # [B, 784]
            xb = xb.view(xb.size(0), -1, 1)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total if total > 0 else 0.0


# =========================
# CSV Export
# =========================

def save_results_to_csv(results, model_paths, filename="accuracy_results_all_datasets.csv"):
    """
    Save results dictionary to CSV file.
    Format: Model, Model_Path, MNIST_Accuracy, EMNIST_Accuracy, PixMix_Accuracy
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["Model", "Model_Path", "MNIST_Accuracy", "EMNIST_Accuracy", "PixMix_Accuracy"]
        writer.writerow(header)
        
        # Data rows
        for model_name, metrics in results.items():
            model_path = model_paths.get(model_name, "N/A")
            row = [
                model_name,
                model_path,
                f"{metrics['MNIST']*100:.4f}",
                f"{metrics['EMNIST']*100:.4f}",
                f"{metrics['PixMix']*100:.4f}"
            ]
            writer.writerow(row)
    
    print(f"\n✅ Results saved to {filename}")


# =========================
# Main
# =========================

def main():
    print("="*60)
    print("Testing Accuracy on All Datasets")
    print("="*60)
    
    # Load all datasets
    try:
        x_mnist, y_mnist = load_mnist_dataset()
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return
    
    try:
        x_emnist, y_emnist = load_emnist_dataset()
    except Exception as e:
        print(f"Error loading EMNIST: {e}")
        return
    
    try:
        x_pixmix, y_pixmix = load_pixmix_dataset()
    except Exception as e:
        print(f"Error loading PixMix: {e}")
        return
    
    results = {}
    
    # Test each model
    for name, ckpt in MODEL_PATHS.items():
        if not os.path.exists(ckpt):
            print(f"⚠ Warning: Missing checkpoint: {ckpt}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        model = load_model(ARCH, ckpt)
        
        # Test on MNIST
        print("Testing on MNIST...")
        acc_mnist = accuracy(model, x_mnist, y_mnist)
        print(f"  MNIST Accuracy: {acc_mnist*100:.2f}%")
        
        # Test on EMNIST
        print("Testing on EMNIST...")
        acc_emnist = accuracy(model, x_emnist, y_emnist)
        print(f"  EMNIST Accuracy: {acc_emnist*100:.2f}%")
        
        # Test on PixMix
        print("Testing on PixMix...")
        acc_pixmix = accuracy(model, x_pixmix, y_pixmix)
        print(f"  PixMix Accuracy: {acc_pixmix*100:.2f}%")
        
        results[name] = {
            'MNIST': acc_mnist,
            'EMNIST': acc_emnist,
            'PixMix': acc_pixmix
        }
    
    # Print summary (transposed: datasets as rows, models as columns)
    print("\n" + "="*60)
    print("Final Results Summary:")
    print("="*60)
    
    # Get all model names
    model_names = list(results.keys())
    
    # Print header with model names
    header = f"{'Dataset':<15}"
    for model_name in model_names:
        header += f" {model_name:<12}"
    print(header)
    print("-" * (15 + 13 * len(model_names)))
    
    # Print each dataset as a row
    datasets = ['MNIST', 'EMNIST', 'PixMix']
    for dataset in datasets:
        row = f"{dataset:<15}"
        for model_name in model_names:
            acc = results[model_name][dataset] * 100
            row += f" {acc:>10.2f}%"
        print(row)
    
    # Save to CSV
    save_results_to_csv(results, MODEL_PATHS)
    
    return results


if __name__ == "__main__":
    main()

