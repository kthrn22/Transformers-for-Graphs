import torch
import numpy as np

def distance(node_pos):
    r"""
    Parameters:
        node_pos (torch.Tensor): 
            3D Coordinates of each node. Shape: [N, 3]

    Returns:
        distance (torch.Tensor):
            3D distance between each pair of nodes. Shape: [N, N]
    """    
    num_nodes = node_pos.shape[0]
    # shape [N, N, 3]
    duplicated_coordinates = torch.broadcast_to(node_pos.unsqueeze(0), (num_nodes, num_nodes, 3))
    # permute to [N, 3, N]
    duplicated_coordinates = duplicated_coordinates.permute(0, 2, 1)

    # shape [N, 3, N]
    distances = torch.broadcast_to(node_pos.unsqueeze(-1), duplicated_coordinates.shape) - duplicated_coordinates
    # shape [N, 3, N] -> shape [N, N]
    distances = torch.sum(distances ** 2, dim = 1) ** 0.5

    return distances