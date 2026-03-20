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
# 1. TRADES adversarial perturbation
# -------------------------------
def trades_perturb(model, images, epsilon=0.01, alpha=0.003, num_steps=10):
    """
    Generate adversarial perturbations by maximizing KL divergence
    between clean and perturbed outputs (TRADES method).
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(images)

    delta = torch.zeros_like(images, requires_grad=True)
    delta.data.uniform_(-epsilon, epsilon)
    delta.data = torch.clamp(images + delta.data, 0, 1) - images

    for _ in range(num_steps):
        adv_logits = model(images + delta)
        loss = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(clean_logits, dim=1),
            reduction="batchmean",
        )
        loss.backward()

        with torch.no_grad():
            delta.data += alpha * delta.grad.data.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images

        delta.grad.zero_()

    model.train()
    return torch.clamp(images + delta.detach(), 0, 1)


# -------------------------------
# 2. TRADES loss
# -------------------------------
def trades_loss(model, images, labels, epsilon, alpha, num_steps, beta):
    """
    Compute TRADES loss: CE(clean) + beta * KL(clean || adv).

    Returns:
        loss: combined TRADES loss
        clean_logits: logits on clean images
        adv_logits: logits on adversarial images
    """
    adv_images = trades_perturb(model, images, epsilon, alpha, num_steps)

    model.train()
    clean_logits = model(images)
    adv_logits = model(adv_images)

    loss_ce = F.cross_entropy(clean_logits, labels)
    loss_kl = F.kl_div(
        F.log_softmax(adv_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction="batchmean",
    )

    loss = loss_ce + beta * loss_kl
    return loss, clean_logits, adv_logits


# -------------------------------
# 3. PGD attack (for evaluation only)
# -------------------------------
def pgd_attack(model, images, labels, epsilon=0.01, alpha=0.003, num_steps=10):
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
# 4. Training Function
# -------------------------------
def train_gtsrb_trades(
    epochs=30,
    batch_size=128,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    adv_epsilon=0.01,
    adv_alpha=0.003,
    adv_steps=10,
    beta=6.0,
    checkpoint_path=None,
    target_acc=None,
    use_scheduler=False,
    clean_finetune_epochs=0,
    clean_finetune_lr=1e-5,
):
    """
    Train GTSRB CNN model using TRADES adversarial training.

    TRADES loss = CE(f(x), y) + beta * KL(f(x) || f(x'))
    where x' maximizes KL divergence from clean predictions.

    Args:
        beta: TRADES regularization weight. Higher = more robust, lower clean acc.
              Typical values: 1.0 (mild), 6.0 (standard), 10.0 (aggressive).
        checkpoint_path: Path to checkpoint file to load.
        target_acc: Target clean test accuracy (0-1). Stops early if reached.
        use_scheduler: Use cosine annealing LR scheduler.
        clean_finetune_epochs: Number of clean-only epochs appended after TRADES
                               training to recover clean accuracy. 0 = disabled.
        clean_finetune_lr: Learning rate for the clean fine-tune phase.
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

    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

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
        total_loss, total_ce, total_kl = 0.0, 0.0, 0.0
        correct, adv_correct, total = 0, 0, 0

        current_lr = optimizer.param_groups[0]['lr']
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            loss, clean_logits, adv_logits = trades_loss(
                model, images, labels,
                epsilon=adv_epsilon, alpha=adv_alpha,
                num_steps=adv_steps, beta=beta,
            )

            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs

            with torch.no_grad():
                clean_preds = clean_logits.argmax(1)
                correct += (clean_preds == labels).sum().item()
                adv_preds = adv_logits.argmax(1)
                adv_correct += (adv_preds == labels).sum().item()
            total += bs

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / total
        adv_train_acc = adv_correct / total
        avg_loss = total_loss / total

        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Adv Train Acc: {adv_train_acc*100:.2f}%"
        )

        clean_acc = evaluate(model, test_loader, device)
        adv_acc = evaluate_adversarial(model, test_loader, device, adv_epsilon, adv_alpha, adv_steps)

        # Track best model
        if clean_acc > best_acc:
            best_acc = clean_acc
            best_path = f"./checkpoints/gtsrb/gtsrb_cnn_trades_eps{adv_epsilon}_beta{beta}_best.pt"
            torch.save(model.state_dict(), best_path)

        # Check if target accuracy is reached
        if target_acc is not None and clean_acc >= target_acc and epoch >= 10:
            print(f"Target accuracy {target_acc*100:.2f}% reached! (Current: {clean_acc*100:.2f}%)")
            print(f"   Stopping training early at epoch {epoch}/{epochs}")
            break

    # -------------------------------
    # Clean fine-tune phase (optional)
    # -------------------------------
    if clean_finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Clean fine-tune phase: {clean_finetune_epochs} epoch(s) at lr={clean_finetune_lr}")
        print(f"{'='*60}\n")
        for param_group in optimizer.param_groups:
            param_group['lr'] = clean_finetune_lr

        for ft_epoch in range(1, clean_finetune_epochs + 1):
            model.train()
            correct, total, total_loss = 0, 0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            print(
                f"[Finetune] Epoch {ft_epoch}/{clean_finetune_epochs} | "
                f"Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%"
            )
            clean_acc = evaluate(model, test_loader, device)

            if clean_acc > best_acc:
                best_acc = clean_acc
                best_path = f"./checkpoints/gtsrb/gtsrb_cnn_trades_eps{adv_epsilon}_beta{beta}_best.pt"
                torch.save(model.state_dict(), best_path)

            if target_acc is not None and clean_acc >= target_acc:
                print(f"Target accuracy {target_acc*100:.2f}% reached during fine-tune!")
                break

    # -------------------------------
    # Save trained model
    # -------------------------------
    save_name = f"gtsrb_cnn_trades_eps{adv_epsilon}_beta{beta}_acc{clean_acc*100:.2f}.pt"
    save_path = f"./checkpoints/gtsrb/{save_name}"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete - model saved to {save_path}")
    print(f"Best clean accuracy during training: {best_acc*100:.2f}%")
    return model


# -------------------------------
# 5. Evaluation (clean)
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
# 6. Evaluation (adversarial)
# -------------------------------
def evaluate_adversarial(model, test_loader, device, epsilon=0.01, alpha=0.003, num_steps=10):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)

        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    adv_acc = correct / total
    print(f"   -> Adversarial Test Accuracy: {adv_acc*100:.2f}%")
    return adv_acc


# -------------------------------
# 7. Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTSRB model with TRADES adversarial training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adv_epsilon", type=float, default=0.01)
    parser.add_argument("--adv_alpha", type=float, default=0.003)
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=6.0,
                        help="TRADES regularization weight (1.0=mild, 6.0=standard, 10.0=aggressive)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load")
    parser.add_argument("--target_acc", type=float, default=None, help="Target test accuracy (0-1). Stops early if reached.")
    parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing LR scheduler")
    parser.add_argument("--clean_finetune_epochs", type=int, default=0,
                        help="Clean-only epochs appended after TRADES to recover clean accuracy (default: 0)")
    parser.add_argument("--clean_finetune_lr", type=float, default=1e-5,
                        help="Learning rate for clean fine-tune phase (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    train_gtsrb_trades(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adv_epsilon=args.adv_epsilon,
        adv_alpha=args.adv_alpha,
        adv_steps=args.adv_steps,
        beta=args.beta,
        checkpoint_path=args.checkpoint,
        target_acc=args.target_acc,
        use_scheduler=args.use_scheduler,
        clean_finetune_epochs=args.clean_finetune_epochs,
        clean_finetune_lr=args.clean_finetune_lr,
    )

    # Example usage:
    # TRADES training from scratch:
    #   python train_gtsrb_trades.py --epochs 60 --adv_epsilon 0.01 --beta 6.0
    # Fine-tune from clean model with lower beta for better clean accuracy:
    #   python train_gtsrb_trades.py --adv_epsilon 0.01 --beta 1.0 --lr 5e-4 --epochs 40 --use_scheduler --checkpoint checkpoints/gtsrb/gtsrb_cnn_eps0.01_acc90.70.pt