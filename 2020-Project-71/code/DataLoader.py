import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307), (0.3081))
                            ]),

def dataload(batch_size=200, transform=transform):
    train_data = datasets.MNIST(root='data', train=True,
                                    download=True, 
                                    transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                                    download=True, transform=transform)

    train_features = torch.FloatTensor(60000, 28, 14)
    train_targets = torch.FloatTensor(60000, 28, 14)

    train_features = (train_data.data[:, :, :14] - 127.5)/127.5
    train_targets = (train_data.data[:, :, 14:] - 127.5)/127.5

    test_features = torch.FloatTensor(60000, 28, 14)
    test_targets = torch.FloatTensor(60000, 28, 14)

    test_features = (test_data.data[:, :, :14] - 127.5)/127.5
    test_targets = (test_data.data[:, :, 14:] - 127.5)/127.5

    import torch.utils.data as data_utils 

    train_data_set = data_utils.TensorDataset(train_features, train_targets)
    test_data_set = data_utils.TensorDataset(test_features, test_targets)

    train_loader = data_utils.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_data_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader