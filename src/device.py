import torch


def best_device():
    """Apple GPU if available, else NVIDIA GPU, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"