from pathlib import Path
import numpy as np

def load_aspect_vectors(path: Path):
    """
    Load aspect vectors from a given path.
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            nid = int(parts[0])
            vector = [float(x) for x in parts[1:]]
            data[nid] = np.array(vector, dtype=np.float32)
    return data