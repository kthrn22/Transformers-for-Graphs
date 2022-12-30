import sys
sys.path.append('./')
sys.path.append('./utils')

import torch
import numpy as np
import heapq
from utils.convert_edge_index_to_adjacency_list import convert_edge_index_to_adjacency_list

def compute_shortest_path_len(root_node, num_nodes, adjacency_list):
    # compute the shortest distance from root node to other nodes in the graph
    # return the paths if needed

    distances = [-1] * num_nodes
    distances[root_node] = 0
    shortest_distances = [(distances[root_node], root_node)]

    while shortest_distances:
        _, u = heapq.heappop(shortest_distances)

        for v, edge_distance, _ in adjacency_list[u]:
            if distances[v] == -1 or distances[u] + edge_distance < distances[v]:
                distances[v] = distances[u] + edge_distance
                heapq.heappush(shortest_distances, (distances[v], v))

    return distances

def trace_shortest_path(root_node, current_node, adjacency_list, distances, max_shortest_len):
    if distances[current_node] == -1 or root_node == current_node:
        return [-1] * max_shortest_len

    trace, shortest_path = [current_node], []
    while trace[-1] != root_node:
        u = trace[-1]
        
        for v, edge_len, edge_id in adjacency_list[u]:
            if distances[u] == distances[v] + edge_len:
                trace.append(v)
                shortest_path.append(edge_id)
                break

    shortest_path += [-1] * (max_shortest_len - len(shortest_path))

    return shortest_path

def shortest_path(num_nodes, edge_index, return_paths: bool = True):
    adjacency_list = convert_edge_index_to_adjacency_list(edge_index, num_nodes)
    
    # 2d array shortest_distances[i][j]: shortest distance from node i to node j, -1 , shape [N, N]
    # if the path from i to j doesnt exist
    shortest_distances = torch.from_numpy(np.array([compute_shortest_path_len(root_node, num_nodes, adjacency_list) 
                                        for root_node in range(num_nodes)])).to(torch.int64)
    max_shortest_len = torch.max(shortest_distances)

    # trace shortest path, shape [N, N, max_shortest_len]
    # shortest_paths[i][j] = [e_1, e_2, ... ,e_n], the shortest path from node i to node j, e_i is the i_th edge's index
    # padded with -1
    if return_paths:
        shortest_paths = np.array([[trace_shortest_path(root_node, node_id, adjacency_list, shortest_distances[root_node], max_shortest_len)
                                for node_id in range(num_nodes)] for root_node in range(num_nodes)])
        shortest_paths = torch.from_numpy(shortest_paths).permute(1, 0, 2)

        return shortest_distances, shortest_paths

    return shortest_distances