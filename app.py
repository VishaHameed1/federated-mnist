import streamlit as st
import torch
import copy
import pandas as pd
import time
from torch.utils.data import DataLoader

import config
from clients.client import Client
from models.cnn import CNN
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.evaluation import evaluate

st.set_page_config(page_title="FL Simulator Dashboard", layout="wide")

st.title("🛡️ Collaborative Federated Learning Simulator")
st.markdown("""
This simulator demonstrates **Privacy-First ML**. Raw data stays on clients; only model weights are shared.
""")

# Setup UI Layout
col_info, col_metrics = st.columns([1, 1])

with col_info:
    st.subheader("⚙️ Current Configuration")
    st.write(f"**Dataset:** `MNIST (Authentic)`")
    st.write(f"**Model:** `CNN`")
    st.write(f"**Clients:** `{config.NUM_CLIENTS}`")
    st.write(f"**Rounds:** `{config.NUM_ROUNDS}`")
    is_iid = st.toggle("IID Distribution", value=True, help="Toggle between IID (randomly shuffled) and Non-IID (sorted by labels) data partitioning.")

with col_metrics:
    st.subheader("🔒 Privacy Proof")
    st.error("Server Raw Data: 0 BYTES")
    weight_info = st.empty()

st.divider()

chart_col, log_col = st.columns([2, 1])
with chart_col:
    st.subheader("Global Model Accuracy")
    acc_chart_placeholder = st.empty()

with log_col:
    st.subheader("Training Logs")
    log_box = st.empty()

if st.button("🚀 Start Simulation"):
    # 1. Load and Partition Data
    train_dataset, test_dataset = load_data()
    client_datasets = partition_data(train_dataset, config.NUM_CLIENTS, iid=is_iid)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Initialize Global Model
    global_model = CNN()
    
    accuracies = []
    total_weight_size = 0

    for r in range(config.NUM_ROUNDS):
        log_box.text(f"Round {r+1}: Training {config.NUM_CLIENTS} clients...")
        client_weights = []
        
        # 3. Local Training (Clients)
        for i in range(config.NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_datasets[i], batch_size=config.BATCH_SIZE, shuffle=True)
            client = Client(i, local_model, loader)
            weights, size = client.train()
            client_weights.append(weights)
            total_weight_size += size

        # 4. Aggregation (Server)
        global_weights = fedavg(client_weights)
        global_model.load_state_dict(global_weights)

        # 5. Evaluate and Update UI
        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        acc_chart_placeholder.line_chart(
            pd.DataFrame(accuracies, columns=["Accuracy"]), 
            width="stretch"
        )
        weight_info.info(f"Total Weights Transferred: {total_weight_size:.2f} KB")
        time.sleep(0.2)

    st.success("✅ Simulation Complete! Privacy preserved via Weight Sharing.")
    st.balloons()