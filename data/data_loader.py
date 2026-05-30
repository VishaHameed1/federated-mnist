import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset

def load_data():
    """Loads authentic MNIST train and test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset

def partition_data(dataset, num_clients, iid=True):
    """Partitions the dataset among clients."""
    total_size = len(dataset)
    if iid:
        # Correctly calculate partition lengths to fix ValueError
        base_size = total_size // num_clients
        lengths = [base_size] * num_clients
        for i in range(total_size % num_clients):
            lengths[i] += 1
            
        return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    else:
        # Authentic label-based Non-IID splitting using torch.chunk
        indices = torch.argsort(dataset.targets)
        return [Subset(dataset, idx) for idx in torch.chunk(indices, num_clients)]