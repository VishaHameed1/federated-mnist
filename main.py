import copy
import os
import torch
from torch.utils.data import DataLoader

import config # Import config
from model import get_model
from clients.client import Client
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.evaluation import evaluate

# Load and partition the dataset for federated training and evaluation
train_dataset, test_dataset = load_data()

# Determine dimensions
if config.DATASET_TYPE == "mnist":
    input_dim, output_dim = 784, 10
else:
    # For TensorDataset
    input_dim = train_dataset.tensors[0].shape[1]
    # Check unique labels for output_dim
    output_dim = len(torch.unique(train_dataset.tensors[1]))

# Initialize global model using factory
global_model = get_model(config.MODEL_TYPE, input_dim, output_dim)

client_datasets = partition_data(train_dataset, config.NUM_CLIENTS, iid=True)
client_loaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True) for ds in client_datasets]

# Create a DataLoader for the global test set
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

for round_num in range(config.NUM_ROUNDS): # Use config.NUM_ROUNDS

    print(
        f"\nRound {round_num+1}"
    )

    client_weights = []

    for client_id in range(config.NUM_CLIENTS): # Use config.NUM_CLIENTS

        local_model = copy.deepcopy(
            global_model
        )

        client = Client(
            client_id,
            local_model,
            client_loaders[client_id]
        )

        weights, _ = client.train() # client.train() now returns weights, size

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

# Save the trained global model
os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
torch.save(global_model.state_dict(), config.MODEL_SAVE_PATH)
print(f"\nGlobal model saved to {config.MODEL_SAVE_PATH}")