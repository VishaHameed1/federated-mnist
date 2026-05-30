import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3
        )

        self.pool = nn.MaxPool2d(
            2, 2
        )

        self.fc1 = nn.Linear(
            16 * 13 * 13,
            10
        )

    def forward(self, x):

        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        return x