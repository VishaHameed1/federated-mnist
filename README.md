# 🛡️ Federated Learning Simulator

A complete, framework-free implementation of Federated Learning for privacy-preserving machine learning.

🎯 **Key Insight**: "Server has ZERO raw data. Only model weights are shared."

## 🏗️ Architecture
```text
                    SERVER
             (Global Model Aggregation)
                   FedAvg Algorithm
                       ↑↓
        ╔──────────────────────────────╗
        │         WEIGHTS ONLY          │  (No raw data!)
        ╚──────────────────────────────╝
        ↑ (Encrypted)  ↑ (Encrypted)  ↑ (Encrypted)
    Hospital 1       Hospital 2       Hospital 3
    (Oncology)      (Cardiology)    (Endocrinology)
```

## 📋 Features

*   **Heterogeneous Training**: Train a single global model using completely different datasets (e.g., Breast Cancer vs. Diabetes) via Feature Padding.
*   **Military-Grade Security**: AES-128 symmetric encryption (Fernet) for all model weights in transit.
*   **Differential Privacy (DP)**: Gaussian noise injection to prevent "Model Inversion" attacks.
*   **Contribution Analysis**: Real-time tracking of which hospital provides the most valuable data to the global model.
*   **Automated Data Pipeline**: Integrated Kaggle API for automatic medical dataset downloads and KNN-based missing value imputation.
*   **Privacy Proof**: Dashboard shows "Server Data: 0 BYTES".
*   **Framework Free**: Built using pure PyTorch and NumPy (no Flower).

## 🚀 Quick Start

### 🐳 Option 1: Docker Deployment (Recommended)
The entire environment is containerized for reproducibility.
```bash
docker-compose up --build
```

### 🐍 Option 2: Native Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Start simulator: `streamlit run app.py`

## 📁 Project Structure

*   `config.py`: Central hub for dataset URLs, model selection, and security toggles.
*   `app.py`: Streamlit dashboard featuring live accuracy charts and security status.
*   `clients/client.py`: Handles local backpropagation, DP noise injection, and weight encryption.
*   `utils/encryption_utils.py`: Logic for Fernet key management and weight serialization.
*   `server/aggregator.py`: Secure FedAvg implementation with decryption/re-encryption layers.
*   `data/data_loader.py`: Handles complex medical data cleaning, imputation, and feature alignment.
*   `model.py`: Library of swappable model architectures.

## 🔐 Security & Privacy Proof

| Security Layer | Method | Benefit |
| :--- | :--- | :--- |
| Data Isolation | Local Training | Raw patient records never leave the hospital. |
| In-Transit Security | AES-128 Encryption | Intercepted weights cannot be read without the Fernet key. |
| Differential Privacy | Gaussian Noise | Prevents leaking individual patient traits through weight changes. |
| Server Hygiene | Zero-Storage | The server holds 0 bytes of raw data at all times. |

## 🧠 Supported Models

| Model | Architecture | Best For |
| :--- | :--- | :--- |
| Simple NN | 3-layer MLP | General classification |
| CNN | Conv + FC | Image/Spatial data |
| Logistic Regression | Linear | Baseline comparison |

## 📊 Supported Datasets

| Dataset | Features | Classes | Healthcare Domain |
| :--- | :--- | :--- | :--- |
| Oncology | 30 | 2 | Breast Cancer Diagnosis |
| Cardiology | 13 | 2 | Heart Disease Prediction |
| Endocrinology | 8 | 2 | Diabetes Outcome Analysis |

## 🔄 How It Works (FedAvg)

1.  Server initializes global weights.
2.  Each client receives global weights and trains locally on private data.
3.  Clients send only updated weights back to the server.
4.  Server averages weights: `θ_global = (θ₁ + θ₂ + ... + θₙ) / n`.
5.  Repeat for multiple rounds.

## 🧬 Logic Verification & Concepts

### 🏥 Simulated Data Silos
Using the `partition_data` logic, the system fragments datasets into isolated local subsets. In Non-IID mode, it proves the system works even when hospitals see completely different distributions of data.

### 🧩 Heterogeneous Data Handling (Padding)
To support hospitals with different data structures (e.g., Hospital A has 30 features, Hospital B has 4), the simulator employs Feature Alignment:

*   **Global Dimension Standardization**: The system uses `GLOBAL_INPUT_DIM` and `GLOBAL_OUTPUT_DIM` from `config.py` to define a "Universal Model Architecture".
*   **Zero Padding**: Datasets with fewer features are padded with zero-tensors to match the `GLOBAL_INPUT_DIM`, ensuring the input shape matches the model's expected weights.
*   **Feature Truncation**: Larger datasets are structured to fit the universal model architecture.
*   **Universal Weights**: This allows a single global model to learn from multiple medical domains simultaneously.

### 💡 Contribution Impact
The simulator evaluates each hospital's update against a global test set before aggregation, identifying which domain-specific silo is most influential to the global model's intelligence.

### 🛡️ Privacy via Weight Sharing
Raw records never leave the client instance. The server aggregates only weight tensors, ensuring the "Zero Raw Data" claim is mathematically enforced.

## 👥 Authors & Roles

*   **Hadiqa Ehsan**: Core Backend Logic (Data Partitioning, FedAvg Aggregator, Client Training, Model Architectures).
*   **Visha Hameed**: Frontend & Integration (Streamlit Dashboard, Configuration System, Main Entry Point).

---

## 📐 Mathematical Appendix: FedAvg Formula

The Federated Averaging algorithm is mathematically defined as:

Given a set of `K` clients, each with `n_k` data samples, the global model weights `w_{t+1}` at round `t+1` are calculated as:

```
w_{t+1} = Σ (n_k / N) * w_{k, t+1}
```

Where:
- `w_{k, t+1}` are the weights of client `k` after local training.
- `n_k` is the number of samples on client `k`.
- `N = Σ n_k` is the total number of samples across all clients.

This weighted average ensures that clients with larger datasets have a proportionally greater influence on the final global model.

---

## 🛠️ Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **ModuleNotFoundError** | Missing dependencies. | Run `pip install -r requirements.txt` again. |
| **High CPU/RAM Usage** | Running CNN with MNIST. | Switch to `SimpleNN` in `config.py` or reduce `BATCH_SIZE`. |
| **Slow Training (Encryption)** | AES-128 overhead. | Set `ENCRYPTION_ENABLED = False` in `config.py` for testing. |
| **NaN Loss** | Learning rate too high. | Reduce `LEARNING_RATE` in `config.py` (e.g., from 0.01 to 0.001). |
| **Kaggle Download Fails** | Missing API credentials. | Ensure `kaggle.json` is in `~/.kaggle/` and permissions are correct. |
| **Gradient Mismatch** | Heterogeneous features. | Verify `GLOBAL_INPUT_DIM` matches the max feature size across datasets. |
