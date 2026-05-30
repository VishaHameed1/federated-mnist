# ⭐ SIMULATOR CONFIGURATION
DATASET_TYPE = "breast_cancer"  # options: "breast_cancer", "iris", "mnist"
MODEL_TYPE = "simple_nn"         # options: "simple_nn", "cnn", "logistic_regression"

# Training Parameters
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080