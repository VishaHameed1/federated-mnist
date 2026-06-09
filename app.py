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
from utils.encryption_utils import decrypt_weights, encrypt_weights, generate_key
st.set_page_config(page_title="FL Simulator Dashboard", layout="wide")

st.title("🛡️ Collaborative Federated Learning Simulator")
st.markdown("""
This simulator demonstrates **Privacy-First ML**. Raw data stays on clients; only model weights are shared.
""")

# Setup UI Layout
col_info, col_metrics = st.columns([1, 1])

with col_info:
    st.subheader("⚙️ Simulation Parameters")

    # Heterogeneous Mode Toggle
    heterogeneous_mode_ui = st.checkbox(
        "Enable Heterogeneous Mode (Multi-Dataset per Hospital)",
        value=getattr(config, "HETEROGENEOUS_MODE", False),
        help="If enabled, each hospital will train on a different dataset as defined in config.py. If disabled, all hospitals train on a single selected dataset."
    )
    config.HETEROGENEOUS_MODE = heterogeneous_mode_ui # Update config

    if heterogeneous_mode_ui:
        st.info("🏥 Edit Hospital-specific Datasets:")
        # Allow interactive editing of the hospital configurations
        config.HOSPITAL_DATA_CONFIGS = st.data_editor(
            config.HOSPITAL_DATA_CONFIGS, 
            num_rows="dynamic",
            use_container_width=True
        )
        config.GLOBAL_INPUT_DIM = st.number_input("Global Input Dimension (Features)", value=config.GLOBAL_INPUT_DIM)
        config.GLOBAL_OUTPUT_DIM = st.number_input("Global Output Dimension (Classes)", value=config.GLOBAL_OUTPUT_DIM)

        model_options = ["simple_nn", "cnn", "logistic_regression"]
        model_type_ui = st.selectbox(
            "Select Model Architecture (for all hospitals)",
            options=model_options,
            index=model_options.index(getattr(config, "MODEL_TYPE", "simple_nn"))
        )
        config.MODEL_TYPE = model_type_ui
    else:
        # Single Dataset Mode
        dataset_options = ["breast_cancer", "mnist", "iris", "heart_disease", "diabetes"] # Extend as needed
        dataset_type_ui = st.selectbox(
            "Select Dataset Type",
            options=dataset_options,
            index=dataset_options.index(getattr(config, "DATASET_TYPE", "breast_cancer"))
        )
        config.DATASET_TYPE = dataset_type_ui
        config.KAGGLE_DATASET_URL = st.text_input("Kaggle Dataset URL", value=config.KAGGLE_DATASET_URL)
        config.CSV_TARGET_COLUMN = st.text_input("CSV Target Column", value=config.CSV_TARGET_COLUMN)

        model_options = ["simple_nn", "cnn", "logistic_regression"]
        model_type_ui = st.selectbox(
            "Select Model Architecture",
            options=model_options,
            index=model_options.index(getattr(config, "MODEL_TYPE", "simple_nn"))
        )
        config.MODEL_TYPE = model_type_ui

    st.markdown("---")
    st.subheader("Training Parameters")
    
    num_clients_ui = st.slider("Number of Clients", 2, 10, config.NUM_CLIENTS)
    config.NUM_CLIENTS = num_clients_ui

    num_rounds_ui = st.slider("Number of Rounds", 1, 20, config.NUM_ROUNDS)
    config.NUM_ROUNDS = num_rounds_ui

    local_epochs_ui = st.slider("Local Epochs per Client", 1, 10, config.LOCAL_EPOCHS)
    config.LOCAL_EPOCHS = local_epochs_ui

    batch_size_ui = st.slider("Batch Size", 16, 128, config.BATCH_SIZE)
    config.BATCH_SIZE = batch_size_ui

    learning_rate_ui = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=config.LEARNING_RATE, format="%.4f")
    config.LEARNING_RATE = learning_rate_ui

    st.markdown("---")
    st.subheader("🔬 Research Settings")
    is_iid = st.toggle("IID Distribution", value=True)
    use_encryption = st.checkbox(
        "Enable Encryption (Weights in Transit)", 
        value=getattr(config, "USE_ENCRYPTION", False), # Use config's default
        help="Symmetric encryption using AES-128 in CBC mode with an HMAC-SHA256 signature for authentication (Fernet standard)."
    )
    config.USE_ENCRYPTION = use_encryption # Update config
    use_dp = st.checkbox("Enable Differential Privacy (Noise)", value=False)
    
    dp_noise_multiplier_ui = st.number_input("DP Noise Multiplier", min_value=0.0, max_value=1.0, value=config.DP_NOISE_MULTIPLIER, format="%.3f")
    config.DP_NOISE_MULTIPLIER = dp_noise_multiplier_ui

    participation_rate_ui = st.slider("Client Participation Rate", 0.1, 1.0, config.CLIENT_PARTICIPATION_RATE)
    config.CLIENT_PARTICIPATION_RATE = participation_rate_ui

    knn_neighbors_ui = st.slider("KNN Neighbors for Imputation", 1, 10, config.KNN_NEIGHBORS)
    config.KNN_NEIGHBORS = knn_neighbors_ui

    with st.expander("🛠️ Advanced & Network Settings"):
        config.ENCRYPTION_KEY = st.text_input("Encryption Key", value=config.ENCRYPTION_KEY, type="password", help="Key for Fernet encryption. Keep this secure.")
        if st.button("🗝️ Generate New Key"):
            new_key = generate_key()
            st.code(new_key)
            st.warning("Save this key! If you encrypt a model and lose the key, you cannot decrypt it.")
        
        config.MODEL_SAVE_PATH = st.text_input("Model Save Path", value=config.MODEL_SAVE_PATH)
        
        srv_col1, srv_col2 = st.columns(2)
        config.SERVER_HOST = srv_col1.text_input("Server Host", value=config.SERVER_HOST)
        config.SERVER_PORT = srv_col2.number_input("Server Port", value=int(config.SERVER_PORT))

    st.markdown("---")
    st.subheader("🧪 Test Saved Model")
    uploaded_model = st.file_uploader("Upload a saved .pth model", type=["pth"])
    if uploaded_model:
        if st.button("Evaluate Uploaded Model"):
            with st.spinner("Loading data and evaluating..."):
                # Data loading logic needs to respect heterogeneous_mode_ui
                if heterogeneous_mode_ui:
                    # For evaluation, we need a test set. Using the first hospital's test set as a proxy.
                    _, test_ds = load_data(
                        config.HOSPITAL_DATA_CONFIGS[0]['type'],
                        config.HOSPITAL_DATA_CONFIGS[0]['url'],
                        config.HOSPITAL_DATA_CONFIGS[0]['target']
                    )
                    in_dim = config.GLOBAL_INPUT_DIM
                    out_dim = config.GLOBAL_OUTPUT_DIM
                else:
                    _, test_ds = load_data(config.DATASET_TYPE) # Use selected dataset_type_ui
                test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
                
                # Determine input/output dims for the architecture
                if not heterogeneous_mode_ui: # Only determine if not in heterogeneous mode, as it's fixed there
                    if hasattr(test_ds, 'tensors') and len(test_ds.tensors) > 0:
                        in_dim = test_ds.tensors[0].shape[1]
                        if len(test_ds.tensors) > 1:
                            out_dim = len(torch.unique(test_ds.tensors[1]))
                        else:
                            out_dim = 1 # Default or error
                    elif hasattr(test_ds, 'data') and hasattr(test_ds, 'targets'): # For torchvision datasets like MNIST
                        if len(test_ds.data.shape) > 2: # Image data (e.g., MNIST)
                            in_dim = test_ds.data.shape[1] * test_ds.data.shape[2] # H*W
                        else: # Tabular data (if data is 2D)
                            in_dim = test_ds.data.shape[1]
                        out_dim = len(test_ds.classes) if hasattr(test_ds, 'classes') else len(torch.unique(torch.tensor(test_ds.targets)))
                    else:
                        st.error("Could not determine input/output dimensions for the selected dataset. Ensure data_loader.py provides a compatible dataset object.")
                        st.stop()

                eval_model = get_model(config.MODEL_TYPE, in_dim, out_dim) # Use selected model_type_ui
                try:
                    loaded_state_dict = torch.load(uploaded_model, map_location="cpu")
                    # If the loaded model was encrypted, it needs to be decrypted first
                    if isinstance(loaded_state_dict, bytes):
                        loaded_state_dict = decrypt_weights(loaded_state_dict, config.ENCRYPTION_KEY)
                    eval_model.load_state_dict(loaded_state_dict)
                    acc = evaluate(eval_model, test_loader)
                    st.success(f"Model Accuracy: {acc:.2f}%")
                except Exception as e:
                    st.error(f"Incompatible model weights or decryption error: {e}")

    st.markdown("---")
    st.subheader("Kaggle Credentials Status")
    if config.KAGGLE_USERNAME and config.KAGGLE_KEY:
        st.success("Kaggle API credentials loaded from .env")
    else:
        st.warning("Kaggle API credentials not found. Automatic dataset download may fail. Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")

with col_metrics:
    st.subheader("📊 Data Distribution")
    dist_chart = st.empty()
    st.error("Server Raw Data: 0 BYTES")
    
    # Visual Security Indicator
    # encryption_tooltip = "Utilizes Fernet: AES-128 in CBC mode with an HMAC-SHA256 signature for data integrity and confidentiality."
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
    # Ensure encryption key is set if encryption is enabled
    if use_encryption and not config.ENCRYPTION_KEY:
        st.error("Encryption is enabled but ENCRYPTION_KEY is not set in config.py or .env. Please set it or disable encryption.")
        st.stop()
    elif use_encryption and config.ENCRYPTION_KEY == "L_W-vToTq9N_54U7AInH7vE3ZfK-8JdFqYp-0H4Yk08=": # Default key
        st.warning("Using default encryption key. For production, generate a new key and set it in your .env file.")

    # 1. Load Data (Heterogeneous or Single)
    client_datasets = [] # Reset client_datasets for each run
    if heterogeneous_mode_ui: # Use the UI value
        st.info("Running in Heterogeneous Mode: Different datasets per hospital.")
        # Validate HOSPITAL_DATA_CONFIGS
        for i, cfg in enumerate(config.HOSPITAL_DATA_CONFIGS):
            if not all(k in cfg and cfg[k] for k in ["type", "url", "target"]):
                st.error(f"Hospital {i+1} configuration is incomplete. Please ensure 'type', 'url', and 'target' are provided for all hospitals.")
                st.stop()

        for cfg in config.HOSPITAL_DATA_CONFIGS:
            train_ds, _ = load_data(cfg['type'], cfg['url'], cfg['target'])
            client_datasets.append(train_ds)
        # Use the first dataset's test set as a proxy or combine them
        _, test_dataset = load_data(
            config.HOSPITAL_DATA_CONFIGS[0]['type'],
            config.HOSPITAL_DATA_CONFIGS[0]['url'],
            config.HOSPITAL_DATA_CONFIGS[0]['target']
        )
        # Determine global input/output dimensions from config for heterogeneous mode
        in_dim = config.GLOBAL_INPUT_DIM
        out_dim = config.GLOBAL_OUTPUT_DIM
    else:
        train_dataset, test_dataset = load_data(config.DATASET_TYPE) # Pass selected dataset_type_ui
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
    
    # Pad shorter arrays with zeros to match the longest array for DataFrame creation
    max_len = max(len(arr) for arr in dist_data)
    padded_dist_data = [np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0) for arr in dist_data]
    df_dist = pd.DataFrame(padded_dist_data).fillna(0)
    dist_chart.bar_chart(df_dist, width="stretch")

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Initialize Global Model
    if not heterogeneous_mode_ui: # Only determine if not in heterogeneous mode, as it's fixed there
        if hasattr(train_dataset, 'tensors') and len(train_dataset.tensors) > 0:
            in_dim = train_dataset.tensors[0].shape[1]
            if len(train_dataset.tensors) > 1:
                out_dim = len(torch.unique(train_dataset.tensors[1]))
            else:
                out_dim = 1 # Default or error
        elif hasattr(train_dataset, 'data') and hasattr(train_dataset, 'targets'): # For torchvision datasets like MNIST
            if len(train_dataset.data.shape) > 2: # Image data (e.g., MNIST)
                in_dim = train_dataset.data.shape[1] * train_dataset.data.shape[2] # H*W
            else: # Tabular data (if data is 2D)
                in_dim = train_dataset.data.shape[1]
            out_dim = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(torch.unique(torch.tensor(train_dataset.targets)))
        else:
            st.error("Could not determine input/output dimensions for the selected dataset. Ensure data_loader.py provides a compatible dataset object.")
            st.stop()

    global_model = get_model(config.MODEL_TYPE, in_dim, out_dim) # Use UI value
    
    accuracies = []
    total_weight_size = 0

    for r in range(config.NUM_ROUNDS): # Use UI value
        status_prefix = "🔒" if use_encryption else "🔓" # Use UI value
        log_box.text(f"{status_prefix} Round {r+1}: Training {config.NUM_CLIENTS} clients...")
        client_weights_payloads = [] # This will hold either state_dicts or encrypted bytes
        contribution_scores = {}
        
        # Research Feature: Client Selection (Partial Participation)
        available_clients = range(config.NUM_CLIENTS)
        selected_clients = np.random.choice(available_clients, 
                                            int(config.NUM_CLIENTS * config.CLIENT_PARTICIPATION_RATE), # Use UI value
                                            replace=False)

        # 3. Local Training (Clients)
        for i in selected_clients:
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_datasets[i], batch_size=config.BATCH_SIZE, shuffle=True) # Use UI value
            client = Client(i, local_model, loader) 
            weights_or_encrypted_bytes, size = client.train(use_dp=use_dp, use_encryption=use_encryption) # Use UI values
            client_weights_payloads.append(weights_or_encrypted_bytes)
            total_weight_size += size

            # Calculate individual contribution impact (How well does THIS hospital's update perform on the global test set?)
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
            if heterogeneous_mode_ui: # Use UI value
                h_name += f" ({config.HOSPITAL_DATA_CONFIGS[i]['type'].upper()})"
            contribution_scores[h_name] = impact_score

        # Update Contribution Chart
        sorted_scores = dict(sorted(contribution_scores.items(), key=lambda item: item[1], reverse=True))
        contribution_chart.bar_chart(pd.Series(sorted_scores), color="#ff4b4b")

        # 4. Aggregation (Server)
        global_weights_payload = fedavg(client_weights_payloads, use_encryption=use_encryption) # Use UI value
        
        # If global weights are encrypted, decrypt them before loading into the model
        if use_encryption: # Use UI value
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
        for i in range(config.NUM_CLIENTS): # Use UI value
            h_loader = DataLoader(client_datasets[i], batch_size=config.BATCH_SIZE) # Use UI value
            h_acc = evaluate(global_model, h_loader)
            h_name = f"Hospital {i+1}"
            if heterogeneous_mode_ui: # Use UI value
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
    if use_encryption: # Use UI value
        torch.save(encrypt_weights(global_model.state_dict(), config.ENCRYPTION_KEY), config.MODEL_SAVE_PATH)
    else:
        torch.save(global_model.state_dict(), config.MODEL_SAVE_PATH)
    st.info(f"Final global model saved to `{config.MODEL_SAVE_PATH}`")

    st.success("✅ Simulation Complete! Privacy preserved via Weight Sharing.")
    st.balloons()