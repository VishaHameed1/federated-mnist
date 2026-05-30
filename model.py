import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.linear(x)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Optimized for small tabular data reshaped or MNIST
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * input_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 2: # Tabular data
            x = x.unsqueeze(1).unsqueeze(2) 
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def get_model(model_type, input_dim, output_dim):
    if model_type == "cnn":
        return CNN(input_dim, output_dim)
    elif model_type == "logistic_regression":
        return LogisticRegression(input_dim, output_dim)
    else:
        return SimpleNN(input_dim, output_dim)