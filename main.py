# diffkd_cifar100.py – Bare‑bones DiffKD implementation for CIFAR‑100 with ResNet teachers/students
# ----------------------------------------------------------------------------------
# Author: ChatGPT (OpenAI o3)
# Date : 2025‑06‑06
# ----------------------------------------------------------------------------------
# This is **NOT** an official reproduction of the NeurIPS 2023 DiffKD paper.  
# It captures the *core* ideas in <300 lines of code so you can run quick
# experiments on CIFAR‑100 with minimal dependencies.  
# ✔️ PyTorch >=2.0  
# ✔️ torchvision >=0.17  
# ❌ fancy schedulers / UNet / EMA / distributed / SLURM scripts
# ----------------------------------------------------------------------------------
from __future__ import annotations

import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

###############################################################################
# 1. Utility helpers
###############################################################################

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top‑k accuracies (returns list)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None, :])
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

###############################################################################
# 2. Tiny Diffusion‑KD module (no DDPM, just one‑shot denoise + linear AE)
###############################################################################
class BottleneckBlock(nn.Module):
    """ResNet bottleneck without downsample (latent C)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 4)
        self.conv2 = nn.Conv2d(channels // 4, channels // 4, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels // 4)
        self.conv3 = nn.Conv2d(channels // 4, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.act(out)


class LightDiffusionKD(nn.Module):
    """Feature‑level DiffKD as a torch Module.

    Steps (simplified):
      1. Compress teacher & student features via 1×1 conv (linear autoencoder enc).
      2. Treat student latent Z_stu as a *noisy* version of Z_tea.
      3. Fuse with Gaussian using learnable γ(x) (Adaptive Noise Matching).
      4. Denoise in one step with 2 bottleneck blocks (Φ).
      5. MSE loss between denoised student & teacher latent.
    """

    def __init__(self, in_channels: int, latent_dim: int = 512):
        super().__init__()
        # linear autoencoder (conv 1×1)
        self.enc = nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False)
        self.dec = nn.Conv2d(latent_dim, in_channels, kernel_size=1, bias=False)

        # light diffusion network Φ
        self.phi = nn.Sequential(
            BottleneckBlock(latent_dim),
            BottleneckBlock(latent_dim),
        )
        # adaptive noise weight predictor (global avg‑pool → fc → σ)
        self.noise_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, f_stu: torch.Tensor, f_tea: torch.Tensor):
        """Return loss and optional debug tensors."""
        # Encode to latent space
        z_tea = self.enc(f_tea).detach()  # stop grad through teacher path
        z_stu = self.enc(f_stu)

        # estimate γ in (0,1)
        gamma = torch.sigmoid(self.noise_weight(f_stu)).view(-1, 1, 1, 1)
        noise = torch.randn_like(z_stu)
        z_noisy = gamma * z_stu + (1.0 - gamma) * noise

        # One‑step denoising (Φ)
        z_denoised = self.phi(z_noisy)

        kd_loss = F.mse_loss(z_denoised, z_tea)
        return kd_loss, {
            "gamma": gamma.mean().item(),
            "kd_loss": kd_loss.item(),
        }

###############################################################################
# 3. Networks – CIFAR‑style ResNets (reshape conv1)
###############################################################################

def resnet_cifar(resnet_fn, num_classes=100):
    model = resnet_fn(weights=None)
    # tweak first conv for 32×32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # fine‑tune final FC
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

###############################################################################
# 4. Training / evaluation
###############################################################################

def train_one_epoch(model_s, model_t, kd_module, loader, optimizer, device, alpha=1.0):
    model_s.train()
    kd_module.train()
    total_loss, total_cls, total_kd = 0.0, 0.0, 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            feats_t, logits_t = forward_with_feats(model_t, imgs)
        feats_s, logits_s = forward_with_feats(model_s, imgs)

        cls_loss = F.cross_entropy(logits_s, targets)
        kd_loss, _ = kd_module(feats_s, feats_t)
        loss = cls_loss + alpha * kd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_cls += cls_loss.item() * imgs.size(0)
        total_kd += kd_loss.item() * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_cls / n, total_kd / n


def evaluate(model, loader, device):
    model.eval()
    top1, top5 = 0.0, 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            top1 += acc1.item() * imgs.size(0)
            top5 += acc5.item() * imgs.size(0)
    n = len(loader.dataset)
    return top1 / n, top5 / n


def forward_with_feats(model, x):
    """Forward pass that also returns last feature map before avg‑pool."""
    feats = []
    def hook(_, __, output):
        feats.append(output)
    h = model.layer4.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    return feats[0], logits

###############################################################################
# 5. Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Bare‑bones DiffKD on CIFAR‑100")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0, help="KD loss weight")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, default=Path("./ckpts"))
    args = parser.parse_args()

    device = torch.device(args.device)

    # Data – standard CIFAR augmentations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # Teacher / Student
    teacher = resnet_cifar(models.resnet34).to(device)
    student = resnet_cifar(models.resnet18).to(device)

    # (Optional) Pre‑train / load teacher weights externally for better results
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    kd_module = LightDiffusionKD(in_channels=512, latent_dim=512).to(device)

    # Optimizer (only student + kd parameters)
    params = list(student.parameters()) + list(kd_module.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e‑4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    best_acc = 0.0
    args.save.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, cls_loss, kd_loss = train_one_epoch(
            student, teacher, kd_module, train_loader, optimizer, device, alpha=args.alpha
        )
        top1, top5 = evaluate(student, test_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.3f} | "
            f"cls {cls_loss:.3f} | kd {kd_loss:.3f} | Top‑1 {top1:.2f} | Top‑5 {top5:.2f}"
        )

        if top1 > best_acc:
            best_acc = top1
            torch.save(student.state_dict(), args.save / "best_student.pth")

    print(f"Best Top‑1: {best_acc:.2f}")


if __name__ == "__main__":
    main()
