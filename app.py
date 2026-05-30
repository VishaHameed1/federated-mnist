import streamlit as st
import torch
import copy
import pandas as pd
import time
from torch.utils.data import DataLoader

from models.cnn import CNN
from clients.client import Client
from server.aggregator import fedavg
from data.data_loader import load_data, partition_data
from utils.evaluation import evaluate

st.set_page_config(page_title="Federated MNIST Dashboard", layout="wide")

st.title("🌐 Federated Learning MNIST Dashboard")
st.markdown("""
This dashboard visualizes the training of a global model across multiple decentralized clients using the **FedAvg** algorithm.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Training Configuration")
num_clients = st.sidebar.selectbox("Number of Clients", options=[3], index=0, help="Non-IID logic is currently optimized for 3 clients.")
num_rounds = st.sidebar.slider("Training Rounds", min_value=1, max_value=20, value=5)
is_iid = st.sidebar.checkbox("IID Distribution", value=False, help="Uncheck for Non-IID (label skew).")

# Placeholder for metrics and charts
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Metrics")
    accuracy_metric = st.empty()
    round_metric = st.empty()
    status_text = st.empty()

with col2:
    st.subheader("Global Model Accuracy")
    chart_placeholder = st.empty()

if st.button("🚀 Start Federated Training"):
    # Initialization
    global_model = CNN()
    accuracies = []
    
    status_text.info("Loading and partitioning data...")
    train_dataset, test_dataset = load_data()
    client_datasets = partition_data(train_dataset, num_clients, iid=is_iid)
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    progress_bar = st.progress(0)
    
    for round_idx in range(num_rounds):
        round_num = round_idx + 1
        round_metric.metric("Current Round", f"{round_num} / {num_rounds}")
        status_text.warning(f"Round {round_num}: Training clients...")
        
        client_weights = []
        
        for client_id in range(num_clients):
            local_model = copy.deepcopy(global_model)
            client = Client(client_id, local_model, client_loaders[client_id])
            weights = client.train()
            client_weights.append(weights)
            
        status_text.warning(f"Round {round_num}: Aggregating weights...")
        global_weights = fedavg(client_weights)
        global_model.load_state_dict(global_weights)
        
        # Evaluation
        accuracy = evaluate(global_model, test_loader)
        accuracies.append(accuracy)
        
        # Update UI
        accuracy_metric.metric("Global Accuracy", f"{accuracy:.2f}%")
        
        # Update Chart
        df_accuracy = pd.DataFrame(accuracies, columns=["Accuracy"])
        chart_placeholder.line_chart(df_accuracy)
        
        # Update Progress
        progress_bar.progress(round_num / num_rounds)
        
    status_text.success("✅ Training Complete!")
    st.balloons()
else:
    st.info("Click 'Start Federated Training' to begin the simulation.")