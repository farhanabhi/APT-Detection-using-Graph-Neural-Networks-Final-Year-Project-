import torch

NODE_TYPES = {
    "user": 0,
    "process": 1,
    "file": 2,
    "ip": 3
}

def encode(node_type):
    type_vec = torch.zeros(4)
    type_vec[NODE_TYPES[node_type]] = 1.0

    stats = torch.rand(4)
    time_feat = torch.rand(2)

    return torch.cat([type_vec, stats, time_feat])
