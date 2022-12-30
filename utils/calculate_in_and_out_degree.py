import torch
import numpy as np

def calculate_in_and_out_degree(edge_index, num_nodes):
    source_index, target_index = edge_index
    out_degree = torch.zeros((num_nodes)).scatter_add_(0, source_index, torch.ones(source_index.shape)).to(torch.long)
    in_degree = torch.zeros((num_nodes)).scatter_add_(0, target_index, torch.ones(target_index.shape)).to(torch.long)

    return in_degree, out_degree