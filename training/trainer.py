import torch

from training.focal_loss import focal_loss
from training.stage_loss import stage_consistency_loss
from model.tc_gat import TCGAT
def train(model, graphs, labels, epochs=20, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        opt.zero_grad()

        # Forward pass
        out, stage_scores = model(graphs)

        # ✅ SEQUENCE-LEVEL LABEL (IMPORTANT FIX)
        seq_label = labels[-1].float().view(1)

        # Classification loss
        loss_cls = focal_loss(
            out.view(1),
            seq_label
        )

        # Stage consistency loss (innovation)
        loss_stage = stage_consistency_loss(stage_scores)

        # Combined loss
        loss = loss_cls + 0.3 * loss_stage

        loss.backward()
        opt.step()

        print(
            f"Epoch {e+1}: "
            f"Total={loss.item():.4f}, "
            f"Cls={loss_cls.item():.4f}, "
            f"Stage={loss_stage.item():.4f}"
        )
        torch.save(model.state_dict(), "results/tcgat_stage_model.pth")
        print("DEBUG shapes:", out.shape, seq_label.shape)