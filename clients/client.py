import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils.encryption_utils import encrypt_weights
import numpy as np

class Client:
    """Represents a decentralized node (e.g., a hospital) that trains locally."""
    def __init__(self, client_id, model, dataloader):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader

    def train(self, use_dp=False, use_encryption=False):
        """Trains the model locally and returns weights + metadata."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE)
        self.model.train()

        for epoch in range(config.LOCAL_EPOCHS):
            for inputs, labels in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Calculate weight size for privacy proof visualization (in KB)
        weights = self.model.state_dict()

        # Research Feature: Differential Privacy (Gaussian Noise Injection)
        if use_dp:
            for key in weights.keys():
                if weights[key].dtype == torch.float32:
                    noise = torch.randn(weights[key].size()) * config.DP_NOISE_MULTIPLIER
                    weights[key] += noise
        
        # Research Feature: Encryption of weights before sending to server
        if use_encryption:
            encrypted_weights = encrypt_weights(weights, config.ENCRYPTION_KEY)
            # The size of the encrypted blob is the actual transmitted size
            weight_size = len(encrypted_weights) / 1024 
            print(f"DEBUG: Client {self.client_id} training complete. Sending {weight_size:.2f} KB of ENCRYPTED weights. Raw data size: {len(self.dataloader.dataset)} samples (STAYING ON CLIENT)")
            return encrypted_weights, weight_size

        weight_size = sum(p.numel() for p in self.model.parameters()) * 4 / 1024 # Assuming float32
        print(f"DEBUG: Client {self.client_id} training complete. Sending {weight_size:.2f} KB of UNENCRYPTED weights. Raw data size: {len(self.dataloader.dataset)} samples (STAYING ON CLIENT)")
        return weights, weight_size