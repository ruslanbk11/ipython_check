from PIL import Image

from torchvision import datasets


class MNISTLeftRightDataset(datasets.MNIST):

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        x = img[:, :14]
        y = img[:, 14:]

        return x, y