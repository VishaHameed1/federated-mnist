import torch
import config
from utils.encryption_utils import decrypt_weights, encrypt_weights

def fedavg(client_weights_list, use_encryption=False):
    """
    Standard Federated Averaging (FedAvg) implementation.
    """
    # Decrypt client weights if encryption is active
    if use_encryption:
        decrypted_weights = []
        for encrypted_bytes in client_weights_list:
            decrypted_weights.append(decrypt_weights(encrypted_bytes, config.ENCRYPTION_KEY))
        client_weights_list = decrypted_weights

    avg_weights = {}
    for key in client_weights_list[0].keys():
        # Stack tensors and calculate mean along the new dimension
        avg_weights[key] = torch.stack([cw[key] for cw in client_weights_list], dim=0).mean(dim=0)
        
    # Re-encrypt aggregated weights if needed
    if use_encryption:
        return encrypt_weights(avg_weights, config.ENCRYPTION_KEY)
    return avg_weights