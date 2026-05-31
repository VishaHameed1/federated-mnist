import copy
import os
import torch
from torch.utils.data import DataLoader

import config # Import config
from model import get_model
from clients.client import Client
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.encryption_utils import decrypt_weights, encrypt_weights
from utils.evaluation import evaluate

# Load and partition the dataset for federated training and evaluation
client_datasets = []
if getattr(config, "HETEROGENEOUS_MODE", False):
    print("Notice: Running in HETEROGENEOUS MODE (Multi-Dataset)")
    for cfg in config.HOSPITAL_DATA_CONFIGS:
        train_ds, _ = load_data(cfg['type'], cfg['url'], cfg['target'])
        client_datasets.append(train_ds)
    # Use first dataset's test set for global evaluation
    _, test_dataset = load_data(
        config.HOSPITAL_DATA_CONFIGS[0]['type'],
        config.HOSPITAL_DATA_CONFIGS[0]['url'],
        config.HOSPITAL_DATA_CONFIGS[0]['target']
    )
else:
    train_dataset, test_dataset = load_data()
    client_datasets = partition_data(train_dataset, config.NUM_CLIENTS, iid=True)

# Determine dimensions
if getattr(config, "HETEROGENEOUS_MODE", False):
    input_dim, output_dim = config.GLOBAL_INPUT_DIM, config.GLOBAL_OUTPUT_DIM
elif config.DATASET_TYPE == "mnist":
    input_dim, output_dim = 784, 10
else:
    input_dim = client_datasets[0].tensors[0].shape[1]
    output_dim = len(torch.unique(client_datasets[0].tensors[1]))

# Initialize global model using factory
global_model = get_model(getattr(config, "MODEL_TYPE", "simple_nn"), input_dim, output_dim)
client_loaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True) for ds in client_datasets]

# Create a DataLoader for the global test set
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

for round_num in range(config.NUM_ROUNDS): # Use config.NUM_ROUNDS

    print(
        f"\nRound {round_num+1}"
    )

    client_weights = []

    for client_id in range(config.NUM_CLIENTS):

        local_model = copy.deepcopy(
            global_model
        )

        client = Client(
            client_id,
            local_model,
            client_loaders[client_id]
        )

        weights_or_encrypted_bytes, _ = client.train(use_encryption=config.USE_ENCRYPTION)

        client_weights.append(
            weights_or_encrypted_bytes
        )

    global_weights = fedavg(
        client_weights,
        use_encryption=config.USE_ENCRYPTION
    )

    # If global_weights are encrypted, decrypt them before loading
    if config.USE_ENCRYPTION:
        decrypted_global_weights = decrypt_weights(global_weights, config.ENCRYPTION_KEY)
        global_model.load_state_dict(
            decrypted_global_weights
        )
    else:
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
if config.USE_ENCRYPTION:
    torch.save(encrypt_weights(global_model.state_dict(), config.ENCRYPTION_KEY), config.MODEL_SAVE_PATH)
else:
    torch.save(global_model.state_dict(), config.MODEL_SAVE_PATH)
print(f"\nGlobal model saved to {config.MODEL_SAVE_PATH}")