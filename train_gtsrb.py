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
from alpha_beta_CROWN.complete_verifier.model_defs import gtsrb_cnn


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
def train_gtsrb(
    epochs=30,
    batch_size=128,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_adversarial=False,
    adv_epsilon=0.01,
    adv_alpha=0.003,
    adv_steps=10,
    checkpoint_path=None,
    target_acc=None,
    mix_ratio=1.0,
    use_scheduler=False,
):
    """
    Train GTSRB CNN model (with optional adversarial training).

    Args:
        checkpoint_path: Path to checkpoint file to load. If None, starts from scratch.
        target_acc: Target test accuracy (0-1). Training stops early if reached. If None, trains for all epochs.
        mix_ratio: Fraction of each batch trained adversarially (0.0=clean, 1.0=all adv). Default 1.0.
        use_scheduler: Use cosine annealing LR scheduler.
    """

    # NOTE: do NOT normalize - the alpha_beta_CROWN model expects raw [0,1] pixel values.
    transform_train = transforms.Compose([
        transforms.Resize((36, 36)),
        transforms.RandomCrop(32),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Datasets and loaders
    data_root = "./datasets/GTSRB"
    train_set = torchvision.datasets.GTSRB(root=data_root, split="train", transform=transform_train, download=True)
    test_set = torchvision.datasets.GTSRB(root=data_root, split="test", transform=transform_test, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # Model and optimizer
    model = gtsrb_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'lr' in checkpoint:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = checkpoint['lr']
                print(f"   Loaded model state (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                print(f"   Loaded model state_dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"   Loaded model state_dict")

        print(f"   -> Resuming training from epoch {start_epoch}")

    os.makedirs("./checkpoints/gtsrb", exist_ok=True)

    # LR scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        for _ in range(1, start_epoch):
            scheduler.step()

    # -------------------------------
    # Training loop
    # -------------------------------
    best_acc = 0.0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        adv_correct = 0

        current_lr = optimizer.param_groups[0]['lr']
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_adversarial:
                n = images.size(0)
                n_adv = int(n * mix_ratio)

                if n_adv > 0 and n_adv < n:
                    # Mixed training: split batch into clean and adversarial
                    clean_images, adv_images_raw = images[n_adv:], images[:n_adv]
                    clean_labels, adv_labels = labels[n_adv:], labels[:n_adv]

                    adv_images = pgd_attack(
                        model, adv_images_raw, adv_labels,
                        epsilon=adv_epsilon, alpha=adv_alpha,
                        num_steps=adv_steps, device=device
                    )

                    all_images = torch.cat([adv_images, clean_images], dim=0)
                    all_labels = torch.cat([adv_labels, clean_labels], dim=0)
                    outputs = model(all_images)
                    loss = criterion(outputs, all_labels)

                    with torch.no_grad():
                        adv_preds = outputs[:n_adv].argmax(1)
                        adv_correct += (adv_preds == adv_labels).sum().item()
                elif n_adv >= n:
                    # Full adversarial training
                    adv_images = pgd_attack(
                        model, images, labels,
                        epsilon=adv_epsilon, alpha=adv_alpha,
                        num_steps=adv_steps, device=device
                    )
                    outputs = model(adv_images)
                    loss = criterion(outputs, labels)

                    with torch.no_grad():
                        adv_preds = outputs.argmax(1)
                        adv_correct += (adv_preds == labels).sum().item()
                else:
                    # Pure clean training (mix_ratio=0)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / total
        avg_loss = total_loss / total

        if use_adversarial:
            adv_acc = adv_correct / total if mix_ratio > 0 else 0
            print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Adv Train Acc: {adv_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        clean_acc = evaluate(model, test_loader, device)
        if use_adversarial:
            adv_acc = evaluate_adversarial(model, test_loader, device, adv_epsilon, adv_alpha, adv_steps)

        # Track best model
        if clean_acc > best_acc:
            best_acc = clean_acc
            best_path = f"./checkpoints/gtsrb/gtsrb_cnn{'_adv' if use_adversarial else ''}_eps{adv_epsilon}_best.pt"
            torch.save(model.state_dict(), best_path)

        # Check if target accuracy is reached
        if target_acc is not None and clean_acc >= target_acc and epoch >= 10:
            print(f"Target accuracy {target_acc*100:.2f}% reached! (Current: {clean_acc*100:.2f}%)")
            print(f"   Stopping training early at epoch {epoch}/{epochs}")
            break

    # -------------------------------
    # Save trained model
    # -------------------------------
    save_name = f"gtsrb_cnn{'_adv' if use_adversarial else ''}_eps{adv_epsilon}_acc{clean_acc*100:.2f}.pt"
    save_path = f"./checkpoints/gtsrb/{save_name}"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete - model saved to {save_path}")
    print(f"Best clean accuracy during training: {best_acc*100:.2f}%")
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
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"   -> Test Accuracy: {acc*100:.2f}%")
    return acc


# -------------------------------
# 4. Evaluation (adversarial)
# -------------------------------
def evaluate_adversarial(model, test_loader, device, epsilon=0.01, alpha=0.003, num_steps=10):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, device)

        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    adv_acc = correct / total
    print(f"   -> Adversarial Test Accuracy: {adv_acc*100:.2f}%")
    return adv_acc


# -------------------------------
# 5. Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTSRB model with optional adversarial training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adv_epsilon", type=float, default=0.01)
    parser.add_argument("--adv_alpha", type=float, default=0.003)
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load")
    parser.add_argument("--target_acc", type=float, default=None, help="Target test accuracy (0-1). Training stops early if reached.")
    parser.add_argument("--mix_ratio", type=float, default=1.0, help="Fraction of batch for adversarial training (0.0=clean, 0.5=mixed, 1.0=all adv)")
    parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing LR scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    train_gtsrb(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_adversarial=args.use_adversarial,
        adv_epsilon=args.adv_epsilon,
        adv_alpha=args.adv_alpha,
        adv_steps=args.adv_steps,
        checkpoint_path=args.checkpoint,
        target_acc=args.target_acc,
        mix_ratio=args.mix_ratio,
        use_scheduler=args.use_scheduler,
    )

    # Example usage:
    # Clean training:
    #   python train_gtsrb.py --epochs 30
    # Adversarial fine-tune from clean model with mixed training:
    #   python train_gtsrb.py --use_adversarial --adv_epsilon 0.01 --epochs 60 --lr 5e-4 --mix_ratio 0.5 --use_scheduler --checkpoint checkpoints/gtsrb/gtsrb_cnn_eps0.01_acc90.70.pt --target_acc 0.907


# CUDA_VISIBLE_DEVICES=0 python train_gtsrb_trades.py \
#   --adv_epsilon 0.01 --adv_alpha 0.003 --adv_steps 10 --beta 6.0 \
#   --epochs 60 --lr 5e-4 --batch_size 128 --use_scheduler \
#   --checkpoint checkpoints/gtsrb/gtsrb_cnn_eps0.01_acc90.16.pt

# # GPU 1: eps=0.02, beta=3.0
# CUDA_VISIBLE_DEVICES=1 /data/judy/conda/envs/alpha-beta-crown/bin/python train_gtsrb_trades.py \
#   --adv_epsilon 0.02 --adv_alpha 0.005 --adv_steps 10 --beta 3.0 \
#   --epochs 60 --lr 5e-4 --batch_size 128 --use_scheduler \
#   --checkpoint checkpoints/gtsrb/gtsrb_cnn_eps0.01_acc90.70.pt

# # GPU 2: eps=0.03, beta=1.0
# CUDA_VISIBLE_DEVICES=2 /data/judy/conda/envs/alpha-beta-crown/bin/python train_gtsrb_trades.py \
#   --adv_epsilon 0.03 --adv_alpha 0.008 --adv_steps 10 --beta 1.0 \
#   --epochs 60 --lr 5e-4 --batch_size 128 --use_scheduler \
#   --checkpoint checkpoints/gtsrb/gtsrb_cnn_eps0.01_acc90.70.pt