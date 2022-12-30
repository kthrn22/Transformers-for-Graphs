import sys
sys.path.append("./")
sys.path.append("./utils")

import torch
import numpy as np
from utils.distance import distances

num_nodes = 5
coordinates = torch.from_numpy(np.random.rand(num_nodes, 3))
print(coordinates)

node_3d_distances = distances(coordinates)
print(node_3d_distances)

print(node_3d_distances[0][1] == torch.sum((coordinates[0] - coordinates[1]) ** 2) ** 0.5)

