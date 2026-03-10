def parse_event(row):
    nodes = {
        "user": row["user"],
        "process": row["process"],
        "file": row["file"],
        "ip": row["ip"]
    }

    edges = [
        (row["user"], row["process"], "login"),
        (row["process"], row["file"], "read"),
        (row["process"], row["ip"], "connect")
    ]

    return nodes, edges
