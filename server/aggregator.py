import torch

def fedavg(client_weights):
    """
    Standard Federated Averaging (FedAvg) implementation.
    """
    avg_weights = {}
    for key in client_weights[0].keys():
        # Stack tensors and calculate mean along the new dimension
        avg_weights[key] = torch.stack([cw[key] for cw in client_weights], dim=0).mean(dim=0)
        
    return avg_weights