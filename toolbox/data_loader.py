import torchvision.transforms as transforms
import torchvision
import torch
import os


def get_loaders(dataset, batch_size):
    if dataset == "Cifar100":
        ds = torchvision.datasets.CIFAR100
        mean, std = (0.5071, 0.4867, 0.4409), (0.267, 0.256, 0.276) 
    elif dataset == "Cifar10":
        ds = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) 

    num_workers = 8 if os.name == 'posix' else 0

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = ds(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = ds(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
    
class DataHelper:
    def __init__(self,dataset,batch_size):
        self.trainloader, self.testloader = get_loaders(dataset, batch_size)
        self.name = dataset
        if dataset == 'Cifar100':
            self.class_num = 100
        elif dataset == 'Cifar10':
            self.class_num = 10

def Cifar10(batch_size=128):
    return DataHelper('Cifar10', batch_size)

def Cifar100(batch_size=128):
    return DataHelper('Cifar100', batch_size)