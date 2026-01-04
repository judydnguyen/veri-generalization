import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add auto-attack to path if it exists locally
_auto_attack_path = os.path.join(os.path.dirname(__file__), "auto-attack")
if os.path.exists(_auto_attack_path) and _auto_attack_path not in sys.path:
    sys.path.insert(0, _auto_attack_path)

from autoattack import AutoAttack
from alpha_beta_CROWN.complete_verifier.model_defs import mnist_fc


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "mnist"          # mnist | fashionmnist
DATA_ROOT = "./data"

ARCH = "mnist_fc"          # same architecture, different weights

# Use relative paths based on script location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "model_A": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc.pt"),
    "model_B": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.03_acc93.17.pt"),
    "model_C": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.01_acc93.60.pt"),
    "model_D": os.path.join(_BASE_DIR, "checkpoints", "mnistfc", "mnist_fc_adv_eps0.05_acc93.24.pt"),
}

BATCH_SIZE = 256
N_EXAMPLES = 1000         # MNIST test set size
NORM = "Linf"              # Linf | L2

# Common MNIST eps (pixel space [0,1])
EPS_LIST = [1/255, 2/255, 3/255, 4/255]


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
# Dataset
# =========================
def load_dataset(name):
    tfm = transforms.ToTensor()  # keep in [0,1]

    if name == "mnist":
        ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    xs, ys = [], []
    seen = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= N_EXAMPLES:
            break

    x_test = torch.cat(xs)[:N_EXAMPLES].to(DEVICE)
    y_test = torch.cat(ys)[:N_EXAMPLES].to(DEVICE)
    return x_test, y_test


# =========================
# Accuracy helpers
# =========================
@torch.no_grad()
def accuracy(model, x, y):
    """
    Compute accuracy. Input x should be in shape [B, 1, 28, 28] or [B, 784].
    Model expects [B, 784, 1] based on training code, so we reshape.
    """
    correct = 0
    for i in range(0, x.size(0), BATCH_SIZE):
        xb = x[i:i+BATCH_SIZE]
        yb = y[i:i+BATCH_SIZE]
        # Reshape to [B, 784, 1] as expected by the model (based on train_mnist.py)
        if xb.dim() == 4:  # [B, 1, 28, 28]
            xb = xb.view(xb.size(0), -1, 1)
        elif xb.dim() == 2:  # [B, 784]
            xb = xb.view(xb.size(0), -1, 1)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / y.numel()


class ModelWrapper(nn.Module):
    """
    Wrapper to make the model compatible with AutoAttack.
    AutoAttack expects standard image format [B, C, H, W] or [B, H*W].
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # AutoAttack may pass [B, C, H, W] or [B, 784]
        if x.dim() == 4:  # [B, 1, 28, 28]
            x = x.view(x.size(0), -1, 1)
        elif x.dim() == 2:  # [B, 784]
            x = x.view(x.size(0), -1, 1)
        return self.model(x)


def robust_accuracy_autoattack(model, x, y, eps):
    # Wrap model for AutoAttack compatibility
    wrapped_model = ModelWrapper(model)
    adversary = AutoAttack(
        wrapped_model,
        norm=NORM,
        eps=eps,
        version="standard",
        device=DEVICE,
    )

    x_adv = adversary.run_standard_evaluation(x, y, bs=BATCH_SIZE)
    return accuracy(model, x_adv, y)


# =========================
# CSV Export
# =========================
def save_results_to_csv(results, model_paths, filename="robustness_results.csv"):
    """
    Save results dictionary to CSV file.
    Format: Model, Model_Path, Clean_Accuracy, eps_1/255, eps_2/255, eps_3/255, eps_4/255
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ["Model", "Model_Path", "Clean_Accuracy"]
        header.extend([f"Robust_Acc_eps{eps:.6f}" for eps in EPS_LIST])
        writer.writerow(header)
        
        # Data rows
        for model_name, metrics in results.items():
            model_path = model_paths.get(model_name, "N/A")
            row = [model_name, model_path, f"{metrics['clean']*100:.4f}"]
            row.extend([f"{metrics[eps]*100:.4f}" for eps in EPS_LIST])
            writer.writerow(row)
    
    print(f"\n✅ Results saved to {filename}")


# =========================
# Main
# =========================
def main():
    x_test, y_test = load_dataset(DATASET)
    results = {}

    for name, ckpt in MODEL_PATHS.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        print(f"\n=== {name} ===")
        model = load_model(ARCH, ckpt)

        clean_acc = accuracy(model, x_test, y_test)
        print(f"Clean Accuracy: {clean_acc*100:.2f}%")

        results[name] = {"clean": clean_acc}

        for eps in EPS_LIST:
            ra = robust_accuracy_autoattack(model, x_test, y_test, eps)
            results[name][eps] = ra
            print(f"  eps={eps:.3f} -> Robust Acc (AA): {ra*100:.2f}%")

    print("\n" + "="*60)
    print("Final results:")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Clean Accuracy: {metrics['clean']*100:.2f}%")
        for eps in EPS_LIST:
            print(f"  eps={eps:.6f}: {metrics[eps]*100:.2f}%")
    
    # Save to CSV
    save_results_to_csv(results, MODEL_PATHS)
    
    return results


if __name__ == "__main__":
    main()