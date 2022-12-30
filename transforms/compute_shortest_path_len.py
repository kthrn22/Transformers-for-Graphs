import torch
import numpy as np
import heapq

def compute_shortest_path_len(root_node, num_nodes, adjacency_list, return_paths: bool = True):
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