import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset

transform = transforms.Compose([
    transforms.ToTensor()
])

def load_data():

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False, # Load the test set
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset # Return both train and test datasets

def partition_data(dataset, num_clients, iid=True):
    if iid:
        partition_size = len(dataset) // num_clients
        partitions = [partition_size] * num_clients
        generator = torch.Generator().manual_seed(42)
        return random_split(dataset, partitions, generator=generator)
    else:
        # Non-IID split: Client 1 (0-3), Client 2 (4-6), Client 3 (7-9)
        labels = dataset.targets
        client_indices = [
            torch.where((labels >= 0) & (labels <= 3))[0],
            torch.where((labels >= 4) & (labels <= 6))[0],
            torch.where((labels >= 7) & (labels <= 9))[0]
        ]
        return [Subset(dataset, idx) for idx in client_indices]