import sys
sys.path.append('./')
sys.path.append('./graph_transformers')

import torch.nn as nn
import torch
import numpy as np
from graph_transformers.Graphormer import GraphormerLayer
from graph_transformers.TransformerM import TransformerMLayer, DistanceEncoding

node_feat = torch.from_numpy(np.random.rand(5, 3)).to(torch.float32)

edge_index = np.array([[0, 1], [1, 0],
                        [0, 2], [2, 0],
                        [0, 3], [3, 0],
                        [0, 4], [4, 0],
                        [1, 2], [2, 1],
                        [1, 4], [4, 1],
                        [2, 3], [3, 2],
                        [2, 4], [4, 2], 
                        [3, 4], [4, 3],]).T
edge_index = torch.from_numpy(edge_index).to(torch.long)
edge_feat = torch.from_numpy(np.random.rand(edge_index.shape[1], 3))
edge_type = torch.tensor([
    [0, 1, 2, 0, 1],
    [1, 2, 0, 1, 2],
    [1, 0, 0, 1, 0],
    [0, 2, 1, 0, 0],
    [2, 1, 0, 1, 2],
])
distances = torch.from_numpy(np.random.rand(5, 5)).to(torch.float32)
# graphormer_layer = GraphormerLayer(node_feat.shape[-1], 128, 3, 0.1)
# out_node_feat = graphormer_layer(node_feat, edge_index, 2, edge_feat)

de = DistanceEncoding(3, 5, 3, 128)
out_node_feat = de(distances, edge_type)

print(out_node_feat)