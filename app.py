import streamlit as st
import os
import torch
import copy
import pandas as pd
import time
import numpy as np
from torch.utils.data import DataLoader

import config
from clients.client import Client
from model import get_model
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
    st.write(f"**Dataset:** `{config.DATASET_TYPE.replace('_', ' ').upper()}`")
    st.write(f"**Model:** `{config.MODEL_TYPE.replace('_', ' ').upper()}`")
    st.write(f"**Clients:** `{config.NUM_CLIENTS}`")
    st.write(f"**Rounds:** `{config.NUM_ROUNDS}`")

    st.markdown("---")
    st.subheader("🔬 Research Settings")
    is_iid = st.toggle("IID Distribution", value=True)
    use_dp = st.checkbox("Enable Differential Privacy (Noise)", value=False)
    participation_rate = st.slider("Client Participation Rate", 0.1, 1.0, 1.0)

    st.markdown("---")
    st.subheader("🧪 Test Saved Model")
    uploaded_model = st.file_uploader("Upload a saved .pth model", type=["pth"])
    if uploaded_model:
        if st.button("Evaluate Uploaded Model"):
            with st.spinner("Loading data and evaluating..."):
                _, test_ds = load_data()
                test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
                
                # Determine input/output dims for the architecture
                if config.DATASET_TYPE == "mnist":
                    in_dim, out_dim = 784, 10
                else:
                    in_dim = test_ds.tensors[0].shape[1]
                    out_dim = len(torch.unique(test_ds.tensors[1]))
                
                eval_model = get_model(config.MODEL_TYPE, in_dim, out_dim)
                try:
                    eval_model.load_state_dict(torch.load(uploaded_model, map_location="cpu"))
                    acc = evaluate(eval_model, test_loader)
                    st.success(f"Model Accuracy: {acc:.2f}%")
                except Exception as e:
                    st.error(f"Incompatible model weights: {e}")

with col_metrics:
    st.subheader("📊 Data Distribution")
    dist_chart = st.empty()
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

    # Visualize Data Distribution per client
    dist_data = []
    for i, ds in enumerate(client_datasets):
        # Handle different dataset types for label access
        if hasattr(train_dataset, 'targets'):
            labels = train_dataset.targets[ds.indices].numpy()
        else:
            labels = train_dataset.tensors[1][ds.indices].numpy()
            
        counts = np.bincount(labels)
        dist_data.append(counts)
    
    df_dist = pd.DataFrame(dist_data).fillna(0)
    dist_chart.bar_chart(df_dist, width="stretch")

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Initialize Global Model
    if config.DATASET_TYPE == "mnist":
        in_dim, out_dim = 784, 10
    else:
        in_dim = train_dataset.tensors[0].shape[1]
        out_dim = len(torch.unique(train_dataset.tensors[1]))

    global_model = get_model(config.MODEL_TYPE, in_dim, out_dim)
    
    accuracies = []
    total_weight_size = 0

    for r in range(config.NUM_ROUNDS):
        log_box.text(f"Round {r+1}: Training {config.NUM_CLIENTS} clients...")
        client_weights = []
        
        # Research Feature: Client Selection (Partial Participation)
        available_clients = range(config.NUM_CLIENTS)
        selected_clients = np.random.choice(available_clients, 
                                            int(config.NUM_CLIENTS * participation_rate), 
                                            replace=False)

        # 3. Local Training (Clients)
        for i in selected_clients:
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_datasets[i], batch_size=config.BATCH_SIZE, shuffle=True)
            client = Client(i, local_model, loader)
            weights, size = client.train(use_dp=use_dp)
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

    # Save the trained global model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(global_model.state_dict(), config.MODEL_SAVE_PATH)
    st.info(f"Final global model saved to `{config.MODEL_SAVE_PATH}`")

    st.success("✅ Simulation Complete! Privacy preserved via Weight Sharing.")
    st.balloons()