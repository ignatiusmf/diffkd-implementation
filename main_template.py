from toolbox.models import ResNet112, ResNet56
from toolbox.data_loader import Cifar100
from toolbox.utils import plot_the_things, evaluate_model

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import tensorly as tl

from pathlib import Path
import argparse

DEVICE = "cuda"
EPOCHS = 150
BATCH_SIZE = 128*4

class GenerationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

# gen_module = GenerationModule(in_channels=64).to(DEVICE)

def FT(x):
    return F.normalize(x.reshape(x.size(0), -1))

def feature_map_distillation(teacher_outputs, student_outputs, targets):
    teacher_fmap = teacher_outputs[2]
    student_fmap = student_outputs[2]

    noise = torch.normal(mean=0.0, std=NOISE_STD, size=teacher_fmap.shape, device=DEVICE)
    noise2 = torch.normal(mean=0.0, std=NOISE_STD, size=teacher_fmap.shape, device=DEVICE)
    if NOISE_TARGET == 'both':
        noisy_fmap_student = student_fmap + noise 
        noisy_fmap_teacher = teacher_fmap + noise2 
        soft_loss = BETA * F.l1_loss(FT(noisy_fmap_teacher), FT(noisy_fmap_student))
    elif NOISE_TARGET == 'student':
        noisy_fmap_student = student_fmap + noise 
        soft_loss = BETA * F.l1_loss(FT(teacher_fmap), FT(noisy_fmap_student))
    elif NOISE_TARGET == 'teacher':
        noisy_fmap_teacher = teacher_fmap + noise 
        soft_loss = BETA * F.l1_loss(FT(noisy_fmap_teacher), FT(student_fmap))

    hard_loss = F.cross_entropy(student_outputs[3], targets)
    return soft_loss, hard_loss

parser = argparse.ArgumentParser(description='Run a training script with custom parameters.')
parser.add_argument('--noise_std', type=float, default='1')
parser.add_argument('--noise_target', type=str, default='student')
parser.add_argument('--experiment_name', type=str, default='no_name')
args = parser.parse_args()

DISTILLATION = feature_map_distillation
NOISE_STD = args.noise_std
NOISE_TARGET = args.noise_target
BETA = 750
EXPERIMENT_PATH = args.experiment_name

Path(f"experiments/{EXPERIMENT_PATH}").mkdir(parents=True, exist_ok=True)
print(vars(args))

# Model setup
model_path = r"toolbox/Cifar100_ResNet112.pth"
teacher = ResNet112(100).to(DEVICE)
teacher.load_state_dict(torch.load(model_path, weights_only=True)["weights"])

student = ResNet56(100).to(DEVICE)

Data = Cifar100(BATCH_SIZE)
trainloader, testloader = Data.trainloader, Data.testloader

optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_hard_loss = []
train_soft_loss = []
train_acc = []
test_loss = []
test_acc = []
max_acc = 0.0

for i in range(EPOCHS):
    print(i)
    teacher.eval()
    student.train()
    total_hard_loss, total_soft_loss, correct, total = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()


        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        if DISTILLATION.__name__ == 'feature_map_distillation':
            soft_loss, hard_loss = DISTILLATION(teacher_outputs, student_outputs, targets)

        loss = soft_loss + hard_loss
        loss.backward()
        optimizer.step()

        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    scheduler.step()

    trhl, trsl = total_hard_loss/(b_idx+1), total_soft_loss/(b_idx+1)
    tra = 100*correct/total

    print(f'TRAIN | Hard Loss: {trhl:.3f} | Soft Loss {trsl:.3f} | Acc: {tra:.3f} |')
    tel, tea = evaluate_model(student, testloader)

    train_hard_loss.append(trhl)
    train_soft_loss.append(trsl)
    train_acc.append(tra)
    test_loss.append(tel)
    test_acc.append(tea)

    if tea > max_acc:
        max_acc = tea
        torch.save({'weights': student.state_dict()}, f'experiments/{EXPERIMENT_PATH}/ResNet56.pth')
    
    plot_the_things((train_hard_loss, train_soft_loss), test_loss, train_acc, test_acc, EXPERIMENT_PATH)

import json

with open(f'experiments/{EXPERIMENT_PATH}/metrics.json', 'w') as f:
    json.dump({
        'train_hard_loss': train_hard_loss,
        'train_soft_loss': train_soft_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, f)