import torch
import torch.nn.functional as F

def focal_loss(pred, target, alpha=0.75, gamma=2):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()
