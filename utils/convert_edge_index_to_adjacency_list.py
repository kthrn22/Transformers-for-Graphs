import torch

def convert_edge_index_to_adjacency_list(edge_index, num_nodes, edge_weights = None):
    adjacency_list = [[] for _ in range(num_nodes)]
    source_vertices, target_vertices = edge_index   
    
    if edge_weights is None:
        edge_weights = torch.ones(edge_index.shape[-1])
    
    for edge_id, (source_vertex, target_vertex, edge_weight) in enumerate(zip(source_vertices, target_vertices, edge_weights)):
        source_vertex, target_vertex, edge_weight = source_vertex.item(), target_vertex.item(), edge_weight.item()

        adjacency_list[source_vertex].append((target_vertex, edge_weight, edge_id))

    return adjacency_list