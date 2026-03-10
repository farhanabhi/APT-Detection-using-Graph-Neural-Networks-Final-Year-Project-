def causal_weight(action, stage):
    if stage != "benign" and action in ["exec", "connect"]:
        return 1.0
    return 0.3
