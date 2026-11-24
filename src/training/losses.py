import torch
import torch.nn as nn


def weighted_ce(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """MVP loss. class_weights index 0 = spoof, 1 = bonafide."""
    return nn.CrossEntropyLoss(weight=class_weights)
