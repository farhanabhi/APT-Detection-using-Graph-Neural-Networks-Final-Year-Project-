import torch
from torch_geometric.nn import GATConv

class HeteroGAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.login = GATConv(in_dim, out_dim)
        self.exec = GATConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros(x.size(0), self.login.out_channels)
        if (edge_type == 0).sum() > 0:
            out += self.login(x, edge_index[:, edge_type == 0])
        if (edge_type == 1).sum() > 0:
            out += self.exec(x, edge_index[:, edge_type == 1])
        return out
