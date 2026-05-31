import torch
from cryptography.fernet import Fernet
import io

def generate_key():
    """Generates a new Fernet key."""
    return Fernet.generate_key().decode('utf-8')

def encrypt_weights(weights_state_dict, key_str):
    """
    Encrypts a PyTorch state_dict.
    The state_dict is first serialized to bytes, then encrypted.
    """
    key = Fernet(key_str.encode('utf-8'))
    
    # Serialize state_dict to bytes
    buffer = io.BytesIO()
    torch.save(weights_state_dict, buffer)
    serialized_weights = buffer.getvalue()
    
    # Encrypt the serialized weights
    encrypted_weights = key.encrypt(serialized_weights)
    return encrypted_weights

def decrypt_weights(encrypted_weights_bytes, key_str):
    """
    Decrypts encrypted bytes back into a PyTorch state_dict.
    """
    key = Fernet(key_str.encode('utf-8'))
    decrypted_weights = key.decrypt(encrypted_weights_bytes)
    buffer = io.BytesIO(decrypted_weights)
    state_dict = torch.load(buffer)
    return state_dict