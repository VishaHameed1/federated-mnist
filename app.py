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
from utils.encryption_utils import decrypt_weights, encrypt_weights
st.set_page_config(page_title="FL Simulator Dashboard", layout="wide")

st.title("🛡️ Collaborative Federated Learning Simulator")
st.markdown("""
This simulator demonstrates **Privacy-First ML**. Raw data stays on clients; only model weights are shared.
""")

# Setup UI Layout
col_info, col_metrics = st.columns([1, 1])

with col_info:
    st.subheader("⚙️ Current Configuration")
    if getattr(config, "HETEROGENEOUS_MODE", False):
        st.warning("🔄 MODE: HETEROGENEOUS (Multi-Dataset)")
        for i, cfg in enumerate(config.HOSPITAL_DATA_CONFIGS):
            st.write(f"**Hospital {i+1}:** `{cfg['type'].upper()}`")
    else:
        st.write(f"**Dataset:** `{config.DATASET_TYPE.replace('_', ' ').upper()}`")
    model_type = getattr(config, "MODEL_TYPE", "simple_nn")
    st.write(f"**Model:** `{model_type.replace('_', ' ').upper()}`")
    st.write(f"**Clients:** `{config.NUM_CLIENTS}`")
    st.write(f"**Rounds:** `{config.NUM_ROUNDS}`")

    st.markdown("---")
    st.subheader("🔬 Research Settings")
    is_iid = st.toggle("IID Distribution", value=True)
    use_encryption = st.checkbox(
        "Enable Encryption (Weights in Transit)", 
        value=getattr(config, "USE_ENCRYPTION", False),
        help="Symmetric encryption using AES-128 in CBC mode with an HMAC-SHA256 signature for authentication (Fernet standard)."
    )
    use_dp = st.checkbox("Enable Differential Privacy (Noise)", value=False)
    participation_rate = st.slider("Client Participation Rate", 0.1, 1.0, 1.0)

    st.markdown("---")
    st.subheader("🧪 Test Saved Model")
    uploaded_model = st.file_uploader("Upload a saved .pth model", type=["pth"])
    if uploaded_model:
        if st.button("Evaluate Uploaded Model"):
            with st.spinner("Loading data and evaluating..."):
                if getattr(config, "HETEROGENEOUS_MODE", False):
                    _, test_ds = load_data(
                        config.HOSPITAL_DATA_CONFIGS[0]['type'],
                        config.HOSPITAL_DATA_CONFIGS[0]['url'],
                        config.HOSPITAL_DATA_CONFIGS[0]['target']
                    )
                else:
                    _, test_ds = load_data()
                test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
                
                # Determine input/output dims for the architecture
                if getattr(config, "HETEROGENEOUS_MODE", False):
                    in_dim = config.GLOBAL_INPUT_DIM
                    out_dim = config.GLOBAL_OUTPUT_DIM
                elif config.DATASET_TYPE == "mnist":
                    in_dim, out_dim = 784, 10
                else:
                    in_dim = test_ds.tensors[0].shape[1]
                    out_dim = len(torch.unique(test_ds.tensors[1]))
                
                eval_model = get_model(getattr(config, "MODEL_TYPE", "simple_nn"), in_dim, out_dim)
                try:
                    loaded_state_dict = torch.load(uploaded_model, map_location="cpu")
                    # If the loaded model was encrypted, it needs to be decrypted first
                    if isinstance(loaded_state_dict, bytes):
                        loaded_state_dict = decrypt_weights(loaded_state_dict, config.ENCRYPTION_KEY)
                    eval_model.load_state_dict(loaded_state_dict)
                    acc = evaluate(eval_model, test_loader)
                    st.success(f"Model Accuracy: {acc:.2f}%")
                except Exception as e:
                    st.error(f"Incompatible model weights: {e}")

with col_metrics:
    st.subheader("📊 Data Distribution")
    dist_chart = st.empty()
    st.error("Server Raw Data: 0 BYTES")
    
    # Visual Security Indicator
    encryption_tooltip = "Utilizes Fernet: AES-128 in CBC mode with an HMAC-SHA256 signature for data integrity and confidentiality."
    if use_encryption:
        st.success("🔒 Network Security: ENCRYPTED")
    else:
        st.warning(
            "🔓 Network Security: UNENCRYPTED", 
            # help="Encryption is disabled. While raw data stays on clients, weights are transmitted without transit encryption." # Removed help parameter
        )

    weight_info = st.empty()
    st.subheader("🏥 Hospital Comparison")
    comparison_table = st.empty()

st.divider()

chart_col, log_col = st.columns([2, 1])
with chart_col:
    st.subheader("Global Model Accuracy")
    acc_chart_placeholder = st.empty()
    st.subheader("💡 Hospital Contribution to Global Accuracy")
    contribution_chart = st.empty()

with log_col:
    st.subheader("Training Logs")
    log_box = st.empty()

if st.button("🚀 Start Simulation"):
    # 1. Load Data (Heterogeneous or Single)
    client_datasets = []
    if getattr(config, "HETEROGENEOUS_MODE", False):
        st.info("Running in Heterogeneous Mode: Different datasets per hospital.")
        for cfg in config.HOSPITAL_DATA_CONFIGS:
            train_ds, _ = load_data(cfg['type'], cfg['url'], cfg['target'])
            client_datasets.append(train_ds)
        # Use the first dataset's test set as a proxy or combine them
        _, test_dataset = load_data(
            config.HOSPITAL_DATA_CONFIGS[0]['type'],
            config.HOSPITAL_DATA_CONFIGS[0]['url'],
            config.HOSPITAL_DATA_CONFIGS[0]['target']
        )
    else:
        train_dataset, test_dataset = load_data()
        client_datasets = partition_data(train_dataset, config.NUM_CLIENTS, iid=is_iid)

    # Visualize Data Distribution per client
    dist_data = []
    for i, ds in enumerate(client_datasets):
        try:
            if hasattr(ds, 'indices'): # Split dataset
                labels = train_dataset.targets[ds.indices].numpy() if hasattr(train_dataset, 'targets') else train_dataset.tensors[1][ds.indices].numpy()
            else: # Full dataset
                labels = ds.tensors[1].numpy() if hasattr(ds, 'tensors') else ds.targets.numpy()
            counts = np.bincount(labels)
            dist_data.append(counts)
        except:
            dist_data.append([len(ds)])
    
    df_dist = pd.DataFrame(dist_data).fillna(0)
    dist_chart.bar_chart(df_dist, width="stretch")

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Initialize Global Model
    if getattr(config, "HETEROGENEOUS_MODE", False):
        in_dim = config.GLOBAL_INPUT_DIM
        out_dim = config.GLOBAL_OUTPUT_DIM
    elif config.DATASET_TYPE == "mnist":
        in_dim, out_dim = 784, 10
    else:
        in_dim = train_dataset.tensors[0].shape[1]
        out_dim = len(torch.unique(train_dataset.tensors[1]))

    global_model = get_model(getattr(config, "MODEL_TYPE", "simple_nn"), in_dim, out_dim)
    
    accuracies = []
    total_weight_size = 0

    for r in range(config.NUM_ROUNDS):
        status_prefix = "🔒" if use_encryption else "🔓"
        log_box.text(f"{status_prefix} Round {r+1}: Training {config.NUM_CLIENTS} clients...")
        client_weights_payloads = [] # This will hold either state_dicts or encrypted bytes
        contribution_scores = {}
        
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
            weights_or_encrypted_bytes, size = client.train(use_dp=use_dp, use_encryption=use_encryption)
            client_weights_payloads.append(weights_or_encrypted_bytes)
            total_weight_size += size

            # Calculate individual contribution impact
            # (How well does THIS hospital's update perform on the global test set?)
            temp_eval_model = copy.deepcopy(global_model)
            
            # Handle state_dict decryption for evaluation
            if use_encryption:
                decrypted_weights_for_eval = decrypt_weights(weights_or_encrypted_bytes, config.ENCRYPTION_KEY)
                temp_eval_model.load_state_dict(decrypted_weights_for_eval)
            else:
                temp_eval_model.load_state_dict(weights_or_encrypted_bytes)

            impact_score = evaluate(temp_eval_model, test_loader) # Evaluate with decrypted weights
            
            h_name = f"Hospital {i+1}"
            if getattr(config, "HETEROGENEOUS_MODE", False):
                h_name += f" ({config.HOSPITAL_DATA_CONFIGS[i]['type'].upper()})"
            contribution_scores[h_name] = impact_score

        # Update Contribution Chart
        sorted_scores = dict(sorted(contribution_scores.items(), key=lambda item: item[1], reverse=True))
        contribution_chart.bar_chart(pd.Series(sorted_scores), color="#ff4b4b")

        # 4. Aggregation (Server)
        global_weights_payload = fedavg(client_weights_payloads, use_encryption=use_encryption)
        
        # If global weights are encrypted, decrypt them before loading into the model
        if use_encryption:
            global_weights = decrypt_weights(global_weights_payload, config.ENCRYPTION_KEY)
        else:
            global_weights = global_weights_payload
        global_model.load_state_dict(global_weights)

        # 5. Evaluate and Update UI
        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        acc_chart_placeholder.line_chart(
            pd.DataFrame(accuracies, columns=["Accuracy"]), 
            width="stretch"
        )

        # Update Hospital Comparison Table
        hospital_stats = []
        for i in range(config.NUM_CLIENTS):
            h_loader = DataLoader(client_datasets[i], batch_size=config.BATCH_SIZE)
            h_acc = evaluate(global_model, h_loader)
            h_name = f"Hospital {i+1}"
            if getattr(config, "HETEROGENEOUS_MODE", False):
                h_name += f" ({config.HOSPITAL_DATA_CONFIGS[i]['type'].upper()})"
            hospital_stats.append({
                "Hospital": h_name, 
                "Local Silo Accuracy": f"{h_acc:.2f}%", 
                "Global Test Accuracy": f"{acc:.2f}%"
            })
        comparison_table.table(pd.DataFrame(hospital_stats))

        weight_info.info(f"Total Weights Transferred: {total_weight_size:.2f} KB")
        time.sleep(0.2)

    # Save the trained global model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    # If encryption is enabled, save the encrypted state_dict
    if use_encryption:
        torch.save(encrypt_weights(global_model.state_dict(), config.ENCRYPTION_KEY), config.MODEL_SAVE_PATH)
    else:
        torch.save(global_model.state_dict(), config.MODEL_SAVE_PATH)
    st.info(f"Final global model saved to `{config.MODEL_SAVE_PATH}`")

    st.success("✅ Simulation Complete! Privacy preserved via Weight Sharing.")
    st.balloons()