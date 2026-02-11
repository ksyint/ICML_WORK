from torchvision import transforms, datasets
from datasets import ChexpertTrainDataset,ChexpertValidationDataset, ChexpertTestDataset, NIHTrainDataset, NIHTestDataset
import torchvision
import torch

def get_mean_var_classes(name):
    name = name.split('_')[-1]
    if name == 'cifar10':
       return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == 'cifar100':
       return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100

    return None


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def get_datasets(name, test = None):
    mean, var, num_classes = get_mean_var_classes(name)
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, var)
            ])
        transform_test = transforms.Compose([transforms.ToTensor()
                                             , transforms.Normalize(mean, var)
                                             ])
        train = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, var)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, var)])
        train = datasets.CIFAR100(root='./data/', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR100(root='./data/', train=False, download=False, transform=transform_test)

    unorm = UnNormalize(mean, var)

    return train, test, num_classes, unorm
