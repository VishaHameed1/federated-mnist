import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, TensorDataset
import opendatasets as od
import os
import json
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import config

def load_data(dataset_type=None, dataset_url=None, target_col=None):
    """Downloads MNIST from Kaggle URL and loads train/test datasets."""
    data_dir = "./data"
    dataset_url = dataset_url or config.KAGGLE_DATASET_URL
    dataset_name = dataset_url.split('/')[-1]
    source_dir = os.path.join(data_dir, dataset_name)
    d_type = dataset_type or config.DATASET_TYPE
    target_column = target_col or config.CSV_TARGET_COLUMN

    # Set Kaggle credentials for automated download via environment variables
    if config.KAGGLE_USERNAME and config.KAGGLE_KEY:
        os.environ['KAGGLE_USERNAME'] = config.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = config.KAGGLE_KEY
        
        # Also create kaggle.json locally to suppress interactive prompts
        with open("kaggle.json", "w") as f:
            # Use the already stripped values from config
            json.dump({"username": config.KAGGLE_USERNAME, "key": config.KAGGLE_KEY}, f)
            
        print(f"DEBUG: Kaggle credentials synced for user: '{config.KAGGLE_USERNAME}'")
    else:
        print("Notice: Kaggle credentials not found in environment. "
              "Interactive login may be required if the dataset isn't already downloaded.")

    if d_type == "mnist":
        mnist_raw_dir = os.path.join(data_dir, "MNIST", "raw")
        # Check if MNIST data exists; if not, download and process from Kaggle
        if not os.path.exists(mnist_raw_dir):
            print(f"Downloading dataset: {dataset_url}")
            od.download(dataset_url, data_dir=data_dir)
            
            os.makedirs(mnist_raw_dir, exist_ok=True)
            for filename in os.listdir(source_dir):
                # Rename files: torchvision expects hyphens, Kaggle often uses dots
                new_name = filename.replace("images.idx3", "images-idx3").replace("labels.idx1", "labels-idx1")
                shutil.move(os.path.join(source_dir, filename), os.path.join(mnist_raw_dir, new_name))
            shutil.rmtree(source_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
        return train_dataset, test_dataset
    else:
        # Generic CSV Logic
        dataset_path = source_dir
        # Check if data exists; if not, download from Kaggle
        if not os.path.exists(dataset_path):
            print(f"Downloading dataset: {dataset_url}")
            try:
                od.download(dataset_url, data_dir=data_dir)
            except Exception as e:
                if "403" in str(e):
                    raise PermissionError(f"Kaggle 403 Forbidden: You must visit {dataset_url} "
                                          "and accept the rules manually in your browser.") from e
                raise e

        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {dataset_path}")
        
        df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
        
        # Drop ID columns often found in Kaggle sets first to clean data before imputation
        # 'Unnamed: 32' is common in Breast Cancer CSVs and is entirely empty
        df = df.drop(columns=['id', 'Id', 'ID', 'Unnamed: 32'], errors='ignore')

        # Handle missing values using KNN Imputation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=config.KNN_NEIGHBORS)
            df.loc[:, numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode target if categorical
        if y.dtype == object or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Select only numeric features for X
        X = X.select_dtypes(include=[np.number]).values
        y = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature Padding: Align to GLOBAL_INPUT_DIM for Heterogeneous training
        def pad_features(data, target_dim):
            if data.shape[1] < target_dim:
                padding = np.zeros((data.shape[0], target_dim - data.shape[1]))
                return np.hstack((data, padding))
            return data[:, :target_dim] # Truncate if too large

        X_train = pad_features(X_train, config.GLOBAL_INPUT_DIM)
        X_test = pad_features(X_test, config.GLOBAL_INPUT_DIM)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.long)
        )
        
        return train_dataset, test_dataset

def partition_data(dataset, num_clients, iid=True):
    """Partitions the dataset among clients."""
    total_size = len(dataset)
    if iid:
        # Correctly calculate partition lengths to fix ValueError
        base_size = total_size // num_clients
        lengths = [base_size] * num_clients
        for i in range(total_size % num_clients):
            lengths[i] += 1
            
        return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    else:
        # Authentic label-based Non-IID splitting using torch.chunk
        if hasattr(dataset, 'targets'): # For MNIST
            labels = dataset.targets
        elif hasattr(dataset, 'tensors'): # For Breast Cancer/Iris
            labels = dataset.tensors[1]
        else:
            # Fallback for Subset or other wrappers
            labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

        # Optimized chunking for Subset compatibility
        indices = torch.argsort(labels)
        return [Subset(dataset, idx.tolist()) for idx in torch.chunk(indices, num_clients)]