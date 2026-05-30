import torch
import torch.nn as nn
import torch.optim as optim

class Client:

    def __init__(
        self,
        client_id,
        model,
        dataloader
    ):

        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader

    def train(self):

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01
        )

        self.model.train()

        for epoch in range(1):

            for images, labels in self.dataloader:

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

        return self.model.state_dict()