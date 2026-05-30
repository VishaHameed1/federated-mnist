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
                  ↑      ↑      ↑
    Client 1         Client 2         Client 3
   (Hospital A)    (Hospital B)    (Hospital C)
```

## 📋 Features
*   **Swappable Datasets**: Breast Cancer, Iris, or MNIST.
*   **Swappable Models**: Simple NN, CNN, or Logistic Regression.
*   **Privacy Proof**: Dashboard shows "Server Data: 0 BYTES".
*   **Framework Free**: Built using pure PyTorch and NumPy (no Flower).

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Simulator**
   Edit `config.py` to change datasets or models.

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
*   `config.py`: Central configuration for the simulator.
*   `app.py`: Streamlit dashboard for training and visualization.
*   `clients/client.py`: Local training logic for decentralized nodes.
*   `server/aggregator.py`: FedAvg implementation.
*   `data/data_loader.py`: Dynamic data loading and partitioning.
*   `model.py`: Library of swappable model architectures.

## 🔐 Privacy Proof
| Location | Data Type | Size |
| :--- | :--- | :--- |
| **Clients** | Raw Data (Private) | ~15 MB |
| **Server** | Model Weights (Public) | ~50 KB |
| **Server** | Raw Data | **0 BYTES** |

## 🧠 Supported Models
| Model | Architecture | Best For |
| :--- | :--- | :--- |
| Simple NN | 3-layer MLP | General classification |
| CNN | Conv + FC | Image/Spatial data |
| Logistic Regression | Linear | Baseline comparison |

## 📊 Supported Datasets
| Dataset | Features | Classes | Use Case |
| :--- | :--- | :--- | :--- |
| Breast Cancer | 30 | 2 | Healthcare privacy |
| Iris | 4 | 3 | Multi-class |
| MNIST | 784 | 10 | Digit recognition |

## 🔄 How It Works (FedAvg)
1. Server initializes global weights.
2. Each client receives global weights and trains locally on private data.
3. Clients send only updated weights back to the server.
4. Server averages weights: `θ_global = (θ₁ + θ₂ + ... + θₙ) / n`.
5. Repeat for multiple rounds.

---
## 👥 Authors & Roles
*   **Visha Hameed**: Data Pipeline, FedAvg Aggregation Logic, Model Architectures.
*   **Hadiqa Ehsan**: Local Client Training Implementation, Evaluation Metrics, Streamlit Dashboard.
