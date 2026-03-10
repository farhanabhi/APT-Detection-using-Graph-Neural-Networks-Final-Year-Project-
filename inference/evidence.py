def extract_evidence(attention_scores, threshold=0.7):
    evidence_edges = []
    for edge, score in attention_scores.items():
        if score > threshold:
            evidence_edges.append(edge)
    return evidence_edges
