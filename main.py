import copy
import torch
from torch.utils.data import DataLoader

from models.cnn import CNN
from clients.client import Client
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.evaluation import evaluate # New import for evaluation

global_model = CNN()

NUM_CLIENTS = 3
ROUNDS = 5
IID = False # Set to False to test Non-IID distribution

# Load and partition the MNIST dataset for federated training and evaluation
train_dataset, test_dataset = load_data() # Unpack both train and test datasets
client_datasets = partition_data(train_dataset, NUM_CLIENTS, iid=IID)
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]

# Create a DataLoader for the global test set
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

    # Evaluate the global model after aggregation
    accuracy = evaluate(global_model, test_loader)
    print(f"Global model accuracy after round {round_num+1}: {accuracy:.2f}%")