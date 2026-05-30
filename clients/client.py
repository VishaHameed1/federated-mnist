import torch
import torch.nn as nn
import torch.optim as optim
import config

class Client:
    """Represents a decentralized node (e.g., a hospital) that trains locally."""
    def __init__(self, client_id, model, dataloader):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader

    def train(self):
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
        weight_size = sum(p.numel() for p in self.model.parameters()) * 4 / 1024
        return weights, weight_size