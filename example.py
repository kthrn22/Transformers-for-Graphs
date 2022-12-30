import torch
import numpy as np
from utils.convert_edge_index_to_adjacency_list import convert_edge_index_to_adjacency_list
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
edge_feat = torch.from_numpy(np.random.rand(edge_index.shape[1], 7))

_ = convert_edge_index_to_adjacency_list(edge_index, 5)
