import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from alpha_beta_CROWN.complete_verifier.model_defs import mnist_fc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# 1. PGD Adversarial Attack
# -------------------------------
def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_steps=10, device="cpu"):
    """
    Generate adversarial examples using Projected Gradient Descent (PGD).
    """
    model.eval()
    delta = torch.zeros_like(images, requires_grad=True)
    delta.data.uniform_(-epsilon, epsilon)
    delta.data = torch.clamp(images + delta.data, 0, 1) - images

    for _ in range(num_steps):
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        with torch.no_grad():
            delta.data += alpha * delta.grad.data.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images

        delta.grad.zero_()

    return torch.clamp(images + delta.detach(), 0, 1)


# -------------------------------
# 2. Training Function
# -------------------------------
def train_mnist(
    epochs=5,
    batch_size=128,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_adversarial=False,
    adv_epsilon=0.1,
    adv_alpha=0.01,
    adv_steps=10,
    checkpoint_path=None,
    target_acc=None,
):
    """
    Train MNIST fully connected model (with optional adversarial training).
    
    Args:
        checkpoint_path: Path to checkpoint file to load. If None, starts from scratch.
                        Can be a state_dict file or a full checkpoint dict with 'state_dict' key.
        target_acc: Target test accuracy (0-1). Training stops early if reached. If None, trains for all epochs.
    """

    # ✅ NOTE: do NOT normalize — the alpha_beta_CROWN model expects raw [0,1] pixel values.
    transform = transforms.ToTensor()

    # Datasets and loaders
    train_set = torchvision.datasets.MNIST(root="./datasets", train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="./datasets", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # Model and optimizer
    model = mnist_fc().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # Full checkpoint with state_dict, optimizer, etc.
                model.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'lr' in checkpoint:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = checkpoint['lr']
                print(f"   ✅ Loaded model state (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                # Assume it's a state_dict directly
                model.load_state_dict(checkpoint)
                print(f"   ✅ Loaded model state_dict")
        else:
            # Assume it's a state_dict directly
            model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded model state_dict")
        
        print(f"   → Resuming training from epoch {start_epoch}")

    os.makedirs("./checkpoints/mnistfc", exist_ok=True)

    # -------------------------------
    # Training loop
    # -------------------------------
    NUM_ADV_EPOCHS = 10 if use_adversarial else 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        adv_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # reshape: [B, 1, 28, 28] → [B, 784, 1]
            images = images.view(images.size(0), -1, 1)

            optimizer.zero_grad()

            if use_adversarial and epoch <= NUM_ADV_EPOCHS:
                adv_images = pgd_attack(
                    model, images, labels,
                    epsilon=adv_epsilon,
                    alpha=adv_alpha,
                    num_steps=adv_steps,
                    device=device
                )
                outputs = model(adv_images)
                loss = criterion(outputs, labels)

                with torch.no_grad():
                    adv_preds = outputs.argmax(1)
                    adv_correct += (adv_preds == labels).sum().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        if use_adversarial:
            adv_acc = adv_correct / total
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Adv Train Acc: {adv_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        clean_acc = evaluate(model, test_loader, device)
        if use_adversarial:
            adv_acc = evaluate_adversarial(model, test_loader, device, adv_epsilon, adv_alpha, adv_steps)
        
        # Check if target accuracy is reached
        if target_acc is not None and clean_acc >= target_acc and epoch >= 10:
            print(f"🎯 Target accuracy {target_acc*100:.2f}% reached! (Current: {clean_acc*100:.2f}%)")
            print(f"   Stopping training early at epoch {epoch}/{epochs}")
            break

    # -------------------------------
    # Save trained model
    # -------------------------------
    save_name = f"mnist_fc{'_adv' if use_adversarial else ''}_eps{adv_epsilon}_acc{clean_acc*100:.2f}.pt"
    save_path = f"./checkpoints/mnistfc/{save_name}"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete — model saved to {save_path}")
    return model


# -------------------------------
# 3. Evaluation (clean)
# -------------------------------
def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1, 1)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"   → Test Accuracy: {acc*100:.2f}%")
    return acc


# -------------------------------
# 4. Evaluation (adversarial)
# -------------------------------
def evaluate_adversarial(model, test_loader, device, epsilon=0.1, alpha=0.01, num_steps=10):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1, 1)
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, device)

        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    adv_acc = correct / total
    print(f"   → Adversarial Test Accuracy: {adv_acc*100:.2f}%")
    return adv_acc


# -------------------------------
# 5. Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST model with optional adversarial training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adv_epsilon", type=float, default=0.1)
    parser.add_argument("--adv_alpha", type=float, default=0.01)
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load")
    parser.add_argument("--target_acc", type=float, default=None, help="Target test accuracy (0-1). Training stops early if reached.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    train_mnist(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_adversarial=args.use_adversarial,
        adv_epsilon=args.adv_epsilon,
        adv_alpha=args.adv_alpha,
        adv_steps=args.adv_steps,
        checkpoint_path=args.checkpoint,
        target_acc=args.target_acc,
    )

    # example usage: python train_mnist.py --use_adversarial --adv_epsilon 0.03 --adv_alpha 0.01 --adv_steps 10 --epochs 40 --batch_size 256 --lr 0.005
    # example usage: python train_mnist.py --checkpoint ./checkpoints/mnistfc/mnist_fc_adv_eps0.1_acc95.23.pt --epochs 10
    # CUDA_VISIBLE_DEVICES=1 python train_mnist.py --checkpoint "./checkpoints/mnistfc/mnist_fc.pt" --use_adversarial --adv_epsilon 0.01 --epochs 20 --target_acc 0.93
    # python train_mnist.py --use_adversarial --adv_epsilon 0.03 --adv_alpha 0.01 --adv_steps 10 --epochs 40 --batch_size 256 --lr 0.005 --target_acc 0.93 --checkpoint "./checkpoints/mnistfc/mnist_fc.pt"