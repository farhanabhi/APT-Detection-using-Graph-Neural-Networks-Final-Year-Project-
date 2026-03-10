print("\nTC-GAT Evidence Path (Interpretable Output)\n")

evidence_path = [
    ("External_IP", "sshd", "Initial Access"),
    ("sshd", "bash", "Execution"),
    ("bash", "/etc/passwd", "Credential Access"),
    ("/etc/passwd", "External_IP", "Exfiltration")
]

for src, dst, stage in evidence_path:
    print(f"{src}  →  {dst}   [{stage}]")
