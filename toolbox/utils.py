import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_model(model, loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[3], targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    print(f'TEST | Loss: {val_loss/(b_idx+1):.3f} | Acc: {100*correct/total:.3f} |')
    return val_loss/(b_idx+1), correct*100/total


def plot_the_things(train_loss, test_loss, train_acc, test_acc, experiment_path):
        plt.plot(np.array(train_loss[0]), linestyle='dotted',color='b', label=f'Train Hard Loss')
        plt.plot(np.array(train_loss[1]), linestyle='dashed',color='b', label=f'Train Soft Loss')
        plt.plot(np.array(test_loss), linestyle='solid',color='b', label=f'Test Loss')

        plt.xlabel('Epoch')
        plt.xlim(0,len(test_loss))
        plt.ylabel('Loss')
    
        plt.legend()
        plt.savefig(f'experiments/{experiment_path}/Loss.png')
        plt.close()

        max_acc = np.max(np.array(test_acc))

        plt.plot(np.array(train_acc), linestyle='dotted',color='r', label=f'Train Accuracy')
        plt.plot(np.array(test_acc), linestyle='solid',color='r', label=f'Test Accuracy')

        plt.xlabel('Epoch')
        plt.xlim(0,len(test_acc))
        plt.ylabel('Accuracy')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 105, 5))
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.axhline(y=max_acc, color='black', linestyle='-', linewidth=0.5)
        plt.text(0, max_acc + 1, f"Max Acc = {max_acc}", color='black', fontsize=8)


        plt.legend()
        plt.savefig(f'experiments/{experiment_path}/Accuracy.png')
        plt.close()