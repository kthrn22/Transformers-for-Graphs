import sys
sys.path.append('./')
sys.path.append('./utils')

import torch
import numpy as np
from utils.calculate_in_and_out_degree import calculate_in_and_out_degree

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

in_degree, out_degree = calculate_in_and_out_degree(edge_index, 5)