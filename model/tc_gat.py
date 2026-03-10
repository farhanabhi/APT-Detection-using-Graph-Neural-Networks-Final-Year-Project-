import torch
import torch.nn as nn


class TCGAT(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()

        # Simple linear projection (safe baseline)
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        # Temporal aggregation
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Binary classifier
        self.classifier = nn.Linear(hidden_dim, 1)

        # 🔥 Innovation: self-supervised stage head
        self.stage_head = nn.Linear(hidden_dim, 1)

    def forward(self, graph_seq):
        """
        graph_seq: list of graph objects
                   each graph must have graph.x (node features)
        """

        embeddings = []
        stage_scores = []

        for g in graph_seq:
            # node features
            x = g.x

            # project nodes
            h = self.node_proj(x)

            # graph-level embedding (mean pooling)
            z = h.mean(dim=0)

            embeddings.append(z)
            stage_scores.append(self.stage_head(z))

        # temporal modeling
        seq = torch.stack(embeddings).unsqueeze(0)
        _, h_n = self.gru(seq)

        final_emb = h_n.squeeze(0)

        y_hat = torch.sigmoid(self.classifier(final_emb))
        return y_hat, stage_scores