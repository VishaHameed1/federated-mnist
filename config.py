import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Ensure .env is loaded from the correct absolute path
    load_dotenv(dotenv_path=Path(__file__).parent / '.env')
except ImportError:
    print("Warning: 'python-dotenv' not installed. Run 'pip install python-dotenv' to automate Kaggle credentials.")


# ⭐ SIMULATOR CONFIGURATION

# Preset 1: MNIST with CNN (Classic Image Federated Learning)
DATASET_TYPE = "mnist"
MODEL_TYPE = "cnn"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/hojjatk/mnist-dataset"
CSV_TARGET_COLUMN = "label"

# Preset 2: Breast Cancer with Logistic Regression (Healthcare Privacy Demo)
# DATASET_TYPE = "breast_cancer"
# MODEL_TYPE = "logistic_regression"
# KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
# CSV_TARGET_COLUMN = "diagnosis"

# Preset 3: Iris with Simple NN (Multi-class Tabular Classification)
# DATASET_TYPE = "iris"
# MODEL_TYPE = "simple_nn"
# KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/uciml/iris"
# CSV_TARGET_COLUMN = "Species"

# Preset 4: Breast Cancer with Simple NN (Deep Learning for Tabular Data)
# DATASET_TYPE = "breast_cancer"
# MODEL_TYPE = "simple_nn"
# KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
# CSV_TARGET_COLUMN = "diagnosis"

# DATASET_TYPE = "mnist"
# MODEL_TYPE = "simple_nn"
# KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/hojjatk/mnist-dataset"
# CSV_TARGET_COLUMN = "diagnosis"

# Kaggle Credentials (Required for automated downloads)
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME").strip() if os.getenv("KAGGL_USERNAME") else None
KAGGLE_KEY = os.getenv("KAGGLE_KEY").strip() if os.getenv("KAGGLE_KEY") else None

# Training Parameters
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Research Features
DP_NOISE_MULTIPLIER = 0.01  # Differential Privacy: Noise added to weights
CLIENT_PARTICIPATION_RATE = 1.0  # Fraction of clients participating per round
KNN_NEIGHBORS = 5  # Number of neighbors for KNN Imputation

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080

# Persistence
MODEL_SAVE_PATH = "results/global_model.pth"