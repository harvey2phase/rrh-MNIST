import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_mnist(train, batch_size, root):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return DataLoader(
        dataset = MNIST(
            root = root,
            download = True,
            train = train,
            transform = img_transform,
        ),
        batch_size = batch_size,
        shuffle = train,
        drop_last = train,
    )

def to_numpy_arrays(dataloader):
    data = list(dataloader)
    data = [[sample[0].numpy(), sample[1].numpy()] for sample in data]
    X = np.vstack([sample[0] for sample in data])
    y = np.hstack([sample[1] for sample in data])
    return X, y

