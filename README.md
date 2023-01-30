# Transformers for Graph
Implementation of Transformers for Graph in Pytorch. 



## Graphormer

### Parameters

* ```in_dim```: int

  Size of each node's input feature vector
* ```hidden_dim```: int
  
  Hidden size of feed forward layers
  
* ```num_heads```: int

  Number of attention heads
  
* ```dropout```: float
  
  Dropout probability
  
### Forward Computation
```
forward(node_feat, edge_index, max_path_len, edge_feat = None, attention_mask = None)
```
* ```node_feat```: torch.tensor

Nodes' features. Shape: [N, in_dim] (N is the number of nodes)

* ```edge_index```: torch.tensor

Shape: [2, E] (E is the number of edges); dtype: torch.long

* ```max_path_len```: int

Maximume number of edges on each path to be encoded.

* ```edge_feat```: None or torch.tensor

Optional. Shape [E, edge_in_dim]

* ```attention_mask```: None or torch.tensor

Shape: [N, N]. Invalid positions are assigned with non-zero values in attention_mask

### Example
```python
import torch.nn as nn
import torch
import numpy as np
from graph_transformers.Graphormer import GraphormerLayer

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

graphormer_layer = GraphormerLayer(node_feat.shape[-1], 128, 3, 0.1)
out_node_feat = graphormer_layer(node_feat, edge_index, 2, edge_feat)

print(out_node_feat)
```
