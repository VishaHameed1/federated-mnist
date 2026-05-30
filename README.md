Excellent. **Federated MNIST with PyTorch** is the project where you move from toy examples to a real Federated Learning system.

This project follows the original FedAvg workflow proposed by Brendan McMahan and colleagues in the foundational Federated Learning paper:

* PMLR Paper: [https://proceedings.mlr.press/v54/mcmahan17a.html](https://proceedings.mlr.press/v54/mcmahan17a.html)

---

# What You'll Build

A Federated Learning system where:

```text
MNIST Dataset
      |
-------------------------
|          |           |
Client1   Client2    Client3
(train)   (train)    (train)
-------------------------
      |
     FedAvg
      |
 Global Model
      |
 Accuracy Evaluation
```

---

# Project Structure

```text
federated-mnist/

│
├── clients/
│   └── client.py
│
├── server/
│   ├── server.py
│   └── aggregator.py
│
├── app.py (Streamlit Dashboard)
│
├── models/
│   └── cnn.py
│
├── data/
│   └── data_loader.py
│
├── utils/
│   └── evaluation.py
│
├── results/
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

---

# Step 1: Install Dependencies

```bash
pip install torch torchvision numpy matplotlib streamlit pandas
```

---

# Step 2: CNN Model

## models/cnn.py

```python
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
```

---

# Step 3: Load and Split MNIST

## data/data_loader.py

```python
import torch
from torchvision import datasets, transforms
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
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset

def partition_data(dataset, num_clients, iid=True):
    if iid:
        partition_size = len(dataset) // num_clients
        partitions = [partition_size] * num_clients
        return random_split(dataset, partitions, generator=torch.Generator().manual_seed(42))
    else:
        labels = dataset.targets
        client_indices = [
            torch.where((labels >= 0) & (labels <= 3))[0],
            torch.where((labels >= 4) & (labels <= 6))[0],
            torch.where((labels >= 7) & (labels <= 9))[0]
        ]
        return [Subset(dataset, idx) for idx in client_indices]
```

---

# Step 4: Create Federated Clients

## clients/client.py

```python
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
```

---

# Step 5: FedAvg Aggregation

## server/aggregator.py

```python
import copy

def fedavg(client_weights):

    avg_weights = copy.deepcopy(
        client_weights[0]
    )

    for key in avg_weights.keys():

        for i in range(
            1,
            len(client_weights)
        ):

            avg_weights[key] += (
                client_weights[i][key]
            )

        avg_weights[key] = (
            avg_weights[key]
            /
            len(client_weights)
        )

    return avg_weights
```

This is the PyTorch implementation of FedAvg.

Reference:
[https://proceedings.mlr.press/v54/mcmahan17a.html](https://proceedings.mlr.press/v54/mcmahan17a.html)

---

# Step 6: Server

## server/server.py

```python
class Server:

    def __init__(self, model):

        self.global_model = model

    def update_model(
        self,
        global_weights
    ):

        self.global_model.load_state_dict(
            global_weights
        )
```

---

# Step 7: Evaluation

## utils/evaluation.py

```python
import torch

def evaluate(
    model,
    testloader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in testloader:

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total
```

---

# Step 8: Main Training Loop

## main.py

```python
import copy
import torch
from torch.utils.data import DataLoader

from models.cnn import CNN
from clients.client import Client
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.evaluation import evaluate

global_model = CNN()

NUM_CLIENTS = 3
ROUNDS = 5
IID = False 

# Data setup
train_dataset, test_dataset = load_data()
client_datasets = partition_data(train_dataset, NUM_CLIENTS, iid=IID)
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for round_num in range(ROUNDS):

    print(
        f"\nRound {round_num+1}"
    )

    client_weights = []

    for client_id in range(NUM_CLIENTS):

        local_model = copy.deepcopy(
            global_model
        )

        client = Client(
            client_id,
            local_model,
            client_loaders[client_id]
        )

        weights = client.train()

        client_weights.append(
            weights
        )

    global_weights = fedavg(
        client_weights
    )

    global_model.load_state_dict(
        global_weights
    )

    print(
        "Aggregation Complete"
    )

    # Evaluation
    accuracy = evaluate(global_model, test_loader)
    print(f"Global model accuracy after round {round_num+1}: {accuracy:.2f}%")
```

---

# Expected Learning Outcomes

After this project you'll understand:

| Concept              | Learned? |
| -------------------- | -------- |
| Client Training      | ✅        |
| Server Aggregation   | ✅        |
| FedAvg               | ✅        |
| Communication Rounds | ✅        |
| Global Model Updates | ✅        |
| PyTorch FL Workflow  | ✅        |
| Model Evaluation     | ✅        |

---

# Suggested Enhancement

After the basic version works, add:

### IID Split

```text
Client1 → random digits
Client2 → random digits
Client3 → random digits
```

### Non-IID Split

```text
Client1 → digits 0-3
Client2 → digits 4-6
Client3 → digits 7-9
```

Then compare accuracy.

This demonstrates one of the most important challenges in Federated Learning and is commonly discussed in FL research.

### Additional References

* PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
* MNIST Database Dataset: [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)
* TorchVision Documentation: [https://pytorch.org/vision/stable/](https://pytorch.org/vision/stable/)

A good next step after this project is building a **Federated MNIST Dashboard** with accuracy graphs, client participation tracking, and round-by-round visualization.
