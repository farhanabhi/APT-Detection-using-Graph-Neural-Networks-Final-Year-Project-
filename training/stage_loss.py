import torch

def stage_consistency_loss(stage_scores):
    """
    Enforces monotonic attack progression:
    stage_score(t+1) >= stage_score(t)
    """
    loss = 0.0
    for i in range(len(stage_scores) - 1):
        loss += torch.relu(stage_scores[i] - stage_scores[i + 1])
    return loss / max(1, len(stage_scores) - 1)