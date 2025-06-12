from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from toolbox.data_loader import Cifar100  
from toolbox.models import ResNet112, ResNet56  
from toolbox.utils import evaluate_model, plot_the_things  

DEVICE = "cuda"
BATCH_SIZE = 128

parser = argparse.ArgumentParser("Run a training script with custom parameters.")
parser.add_argument("--experiment_name", default="test", type=str)
args = parser.parse_args()
print("\nConfig:", vars(args))

EPOCHS = 150
DIFF_EPOCHS = 3 

EXPERIMENT_PATH = args.experiment_name
Path(f"experiments/{EXPERIMENT_PATH}").mkdir(parents=True, exist_ok=True)


class LightDiffusion(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="mean")


teacher = ResNet112(100).to(DEVICE)
teacher.load_state_dict(torch.load("toolbox/Cifar100_ResNet112.pth", weights_only=True)["weights"])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = ResNet56(100).to(DEVICE)

Data = Cifar100(BATCH_SIZE)
trainloader, testloader = Data.trainloader, Data.testloader


latent_channels = 64  
phi = LightDiffusion(latent_channels).to(DEVICE)

optim_phi = optim.SGD(phi.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched_phi = optim.lr_scheduler.CosineAnnealingLR(optim_phi, T_max=DIFF_EPOCHS)

print(f"\n[Stage A] Pre-training diffusion head for {DIFF_EPOCHS} epochs…")
for epoch in range(DIFF_EPOCHS):
    phi.train()
    running_loss = 0.0

    for inputs, _ in trainloader:
        inputs = inputs.to(DEVICE)

        
        with torch.no_grad():
            t_feat = teacher(inputs)[2].detach()  

        
        eps = torch.randn_like(t_feat)
        noisy_t = t_feat + eps
        pred_eps = phi(noisy_t)
        loss = mse(pred_eps, eps)

        optim_phi.zero_grad()
        loss.backward()
        optim_phi.step()

        running_loss += loss.item()

    sched_phi.step()
    print(f"Epoch {epoch+1:03d}/{DIFF_EPOCHS}  |  Ldiff={running_loss/len(trainloader):.4f}")


for p in phi.parameters():
    p.requires_grad_(False)
phi.eval()


torch.save({"weights": phi.state_dict()},  f"experiments/{EXPERIMENT_PATH}/phi_pretrained.pth")




print(f"\n[Stage B] Training student for {EPOCHS} epochs …")
optim_stu = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched_stu = optim.lr_scheduler.CosineAnnealingLR(optim_stu, T_max=EPOCHS)


train_hard, train_kd, train_acc = [], [], []
val_loss, val_acc = [], []
best_acc = 0.0

for epoch in range(EPOCHS):
    student.train()
    running_hard, running_kd, correct, seen = 0.0, 0.0, 0, 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        
        with torch.no_grad():
            t_feat, _, t_logits = teacher(inputs)[2], None, teacher(inputs)[3]
            t_feat = t_feat.detach()

        s_feat, _, s_logits = student(inputs)[2], None, student(inputs)[3]

        
        eps_s = phi(s_feat)  
        denoised_s = s_feat - eps_s
        kd_loss = mse(denoised_s, t_feat)

        hard_loss = F.cross_entropy(s_logits, targets)
        loss = hard_loss + 1.0 * kd_loss

        optim_stu.zero_grad()
        loss.backward()
        optim_stu.step()

        
        running_hard += hard_loss.item()
        running_kd += kd_loss.item()
        _, pred = s_logits.max(1)
        correct += pred.eq(targets).sum().item()
        seen += targets.size(0)

    sched_stu.step()

    epoch_hard = running_hard / len(trainloader)
    epoch_kd = running_kd / len(trainloader)
    epoch_acc = 100.0 * correct / seen
    train_hard.append(epoch_hard)
    train_kd.append(epoch_kd)
    train_acc.append(epoch_acc)

    print(f"Epoch {epoch+1:03d}/{EPOCHS}  |  Hard={epoch_hard:.3f}  KD={epoch_kd:.3f}  Acc={epoch_acc:.2f}%")

    
    v_loss, v_acc = evaluate_model(student, testloader)
    val_loss.append(v_loss)
    val_acc.append(v_acc)

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save({'weights': student.state_dict()}, f'experiments/{EXPERIMENT_PATH}/ResNet56.pth')

    
    plot_the_things((train_hard, train_kd), val_loss, train_acc, val_acc, EXPERIMENT_PATH)


with open(f'experiments/{EXPERIMENT_PATH}/metrics.json', 'w') as f:
    json.dump({
        "train_hard_loss": train_hard,
        "train_kd_loss": train_kd,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, f)

print(f"\nTraining finished. Best top‑1 acc: {best_acc:.2f}%  (weights saved)")
