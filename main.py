from pathlib import Path
import argparse
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from toolbox.data_loader import Cifar100
from toolbox.models import ResNet112, ResNet56
from toolbox.utils import evaluate_model, plot_the_things

# ----------------------------
# Utilities
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffConfig:
    T_train: int = 1000           # diffusion training steps
    t_start: int = 500            # starting timestep for reverse denoise (as in paper Appx. B.1)
    nfe: int = 5                  # DDIM steps
    lambda_diff: float = 1.0      # weight for Ldiff (teacher feature noise prediction)
    lambda_kd: float = 1.0        # weight for Ldiffkd (feature MSE on denoised student)
    lambda_task: float = 1.0      # weight for CE task loss
    tau_kd: float = 1.0           # temperature for KL on logits (paper default 1)

CFG = DiffConfig()

# ----------------------------
# Time embedding (sinusoidal) & schedule
# ----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):  # t in [0, T)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, -10, half, device=device)
        )  # simple exponential range
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class Bottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 8)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.act(out + identity)
        return out

class EfficientDenoiser(nn.Module):
    """Light-weight diffusion model (2 bottlenecks) with FiLM from time embedding.
    Predicts eps given z_t and t.
    """
    def __init__(self, channels: int, t_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.ReLU(inplace=True),
            nn.Linear(t_dim, 2 * channels)  # scale,bias
        )
        self.block1 = Bottleneck(channels)
        self.block2 = Bottleneck(channels)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, zt, t):
        # FiLM conditioning
        tb = self.time_mlp(t)
        scale, bias = torch.chunk(tb, 2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        x = zt * (1 + scale) + bias
        x = self.block1(x)
        x = self.block2(x)
        eps = self.out(x)
        return eps

class NoiseAdapter(nn.Module):
    """Learns gamma in [0,1] per-sample to match student feature to init diffusion level."""
    def __init__(self, channels: int):
        super().__init__()
        red = max(8, channels // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(red, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):  # x: [B,C,H,W]
        g = self.fc(self.gap(x))  # [B,1,1,1] in (0,1)
        return g

# Precompute linear beta schedule
class DiffusionSchedule:
    def __init__(self, T: int, device: str):
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        self.ab = torch.cumprod(alphas, dim=0)          # alpha_bar_t
        self.sqrt_ab = torch.sqrt(self.ab)
        self.sqrt_1mab = torch.sqrt(1.0 - self.ab)
        self.T = T

    def sample_z_t(self, z0, t):
        eps = torch.randn_like(z0)
        a = self.sqrt_ab[t].view(-1, 1, 1, 1)
        b = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        zt = a * z0 + b * eps
        return zt, eps

    def ddim_step(self, zt, t, t_prev, eps_pred):
        """Deterministic DDIM update (eta=0). t and t_prev are int tensors of same shape.
        z_{t-Δ} = sqrt(ab_{t-Δ}) * x0 + sqrt(1 - ab_{t-Δ}) * eps_pred
        where x0 = (zt - sqrt(1-ab_t)*eps_pred)/sqrt(ab_t)
        """
        ab_t = self.ab[t].view(-1, 1, 1, 1)
        ab_tp = self.ab[t_prev].view(-1, 1, 1, 1)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_1mab_t = torch.sqrt(1.0 - ab_t)
        x0 = (zt - sqrt_1mab_t * eps_pred) / (sqrt_ab_t + 1e-8)
        z_prev = torch.sqrt(ab_tp) * x0 + torch.sqrt(1.0 - ab_tp) * eps_pred
        return z_prev

# ----------------------------
# CLI & experiment setup
# ----------------------------
parser = argparse.ArgumentParser("Run DiffKD on CIFAR-100 (ResNet112->ResNet56)")
parser.add_argument("--experiment_name", default="diffkd_fixed", type=str)
args = parser.parse_args()
print("\nConfig:", vars(args))

EPOCHS = 150
BATCH_SIZE = 128
Path(f"experiments/{args.experiment_name}").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Teacher / Student
# ----------------------------
teacher = ResNet112(100).to(DEVICE)
teacher.load_state_dict(torch.load("toolbox/Cifar100_ResNet112.pth", weights_only=True)["weights"])  # type: ignore
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = ResNet56(100).to(DEVICE)

Data = Cifar100(BATCH_SIZE)
trainloader, testloader = Data.trainloader, Data.testloader

# Get feature channels from a dummy forward
with torch.no_grad():
    tmp = torch.randn(2, 3, 32, 32, device=DEVICE)
    feat_ch = teacher(tmp)[2].shape[1]

phi = EfficientDenoiser(channels=feat_ch).to(DEVICE)
adapter = NoiseAdapter(channels=feat_ch).to(DEVICE)
sched = DiffusionSchedule(CFG.T_train, DEVICE)

# CIFAR: we skip the linear autoencoder (as in Appx. B.2) and work directly on the feature map.

optim_all = optim.SGD(
    [
        {"params": student.parameters(), "lr": 0.1},
        {"params": phi.parameters(), "lr": 0.1},
        {"params": adapter.parameters(), "lr": 0.1},
    ],
    momentum=0.9, weight_decay=5e-4
)
sched_all = optim.lr_scheduler.CosineAnnealingLR(optim_all, T_max=EPOCHS)

# ----------------------------
# Training
# ----------------------------
train_hard, train_kd, train_diff, train_acc = [], [], [], []
val_loss, val_acc = [], []
best_acc = 0.0

for epoch in range(EPOCHS):
    student.train(); phi.train(); adapter.train()
    running_hard = running_kd = running_diff = 0.0
    correct = seen = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_outs = teacher(inputs)
            t_feat = t_outs[2].detach()          # [B,C,H,W]
            t_logits = t_outs[3].detach()        # [B,num_classes]

        # Student forward (we need its feature + logits)
        s_outs = student(inputs)
        s_feat = s_outs[2]
        s_logits = s_outs[3]

        # --------- (A) Train diffusion model on teacher features  ---------
        B = t_feat.size(0)
        t = torch.randint(low=1, high=CFG.T_train, size=(B,), device=DEVICE)
        zt, eps = sched.sample_z_t(t_feat, t)
        eps_pred = phi(zt, t)
        Ldiff = F.mse_loss(eps_pred, eps, reduction="mean")

        # --------- (B) Denoise student feature with DDIM (phi frozen for this path) ---------
        with torch.no_grad():
            gamma = adapter(s_feat)  # [B,1,1,1] in (0,1)
            eps_T = torch.randn_like(s_feat)
            z_t = gamma * s_feat + (1 - gamma) * eps_T  # Eq. (8)

            # choose NFE indices from t_start -> 0
            step_idxs = torch.linspace(CFG.t_start, 0, CFG.nfe+1, dtype=torch.long, device=DEVICE)
            t_vec = step_idxs[:-1].repeat(B)
            tprev_vec = step_idxs[1:].repeat(B)
            # batched deterministic DDIM
            for k in range(CFG.nfe):
                cur_t = torch.full((B,), int(step_idxs[k].item()), device=DEVICE, dtype=torch.long)
                eps_pred_s = phi(z_t, cur_t)
                z_t = sched.ddim_step(z_t, cur_t, torch.full((B,), int(step_idxs[k+1].item()), device=DEVICE, dtype=torch.long), eps_pred_s)
            s_feat_denoised = z_t

        # Feature-level KD (MSE on denoised student vs clean teacher latent)
        Lkd_feat = F.mse_loss(s_feat_denoised, t_feat, reduction="mean")

        # Logits KD (KL with T=1, teacher as target)
        Ttau = CFG.tau_kd
        if Ttau != 1.0:
            s_logit_t = s_logits / Ttau
            t_logit_t = t_logits / Ttau
            Lkd_logit = F.kl_div(F.log_softmax(s_logit_t, dim=1), F.softmax(t_logit_t, dim=1), reduction="batchmean") * (Ttau**2)
        else:
            Lkd_logit = F.kl_div(F.log_softmax(s_logits, dim=1), F.softmax(t_logits, dim=1), reduction="batchmean")

        # Supervised CE on labels
        Ltask = F.cross_entropy(s_logits, targets)

        loss = CFG.lambda_task * Ltask + CFG.lambda_kd * (Lkd_feat + Lkd_logit) + CFG.lambda_diff * Ldiff

        optim_all.zero_grad()
        loss.backward()
        optim_all.step()

        running_hard += Ltask.item()
        running_kd += (Lkd_feat.item() + Lkd_logit.item())
        running_diff += Ldiff.item()

        with torch.no_grad():
            _, pred = s_logits.max(1)
            correct += pred.eq(targets).sum().item()
            seen += targets.size(0)

    sched_all.step()

    epoch_hard = running_hard / len(trainloader)
    epoch_kd = running_kd / len(trainloader)
    epoch_diff = running_diff / len(trainloader)
    epoch_acc = 100.0 * correct / max(seen,1)

    train_hard.append(epoch_hard)
    train_kd.append(epoch_kd)
    train_diff.append(epoch_diff)
    train_acc.append(epoch_acc)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | CE={epoch_hard:.3f}  KD={epoch_kd:.3f}  Ldiff={epoch_diff:.3f}  Acc={epoch_acc:.2f}%")

    # ----- validation -----
    v_loss, v_acc = evaluate_model(student, testloader)
    val_loss.append(v_loss)
    val_acc.append(v_acc)

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save({'weights': student.state_dict()}, f'experiments/{args.experiment_name}/ResNet56.pth')

    plot_the_things((train_hard, train_kd, train_diff), val_loss, train_acc, val_acc, args.experiment_name)

with open(f'experiments/{args.experiment_name}/metrics.json', 'w') as f:
    json.dump({
        "train_hard_loss": train_hard,
        "train_kd_loss": train_kd,
        "train_diff_loss": train_diff,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, f)

print(f"\nTraining finished. Best top1 acc: {best_acc:.2f}%  (weights saved)")
