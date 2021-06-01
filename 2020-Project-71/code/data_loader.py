import torch.utils.data as data_utils
from torchvision import transforms

from datasets import MNISTLeftRightDataset


def get_data_loader(type, train, batch_size):
    if type == 'mnist_left_right':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = MNISTLeftRightDataset(root='data/', train=train, download=True, transform=transform)
        loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError('data_type is unknown: {}'.format(type))
    return loader