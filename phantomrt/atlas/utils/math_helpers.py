"""Math helper functions."""
import torch
import numpy as np


def stable_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithm with numerical stability."""
    return torch.log(x + eps)


def stable_exp(x: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    """Exponential with clamping to prevent overflow."""
    return torch.exp(torch.clamp(x, max=max_val))


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def exponential_moving_average(current: float, new: float, alpha: float = 0.1) -> float:
    """EMA update."""
    return alpha * new + (1 - alpha) * current
