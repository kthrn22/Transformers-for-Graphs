import sys
sys.path.append("./")
sys.path.append("./utils")
sys.path.append("./transforms")

import torch
import torch.nn as nn
import numpy as np
from transforms.shortest_path import shortest_path

# # Utils
# def calculate_in_and_out_degree(edge_index, num_nodes):
#     source_index, target_index = edge_index
#     out_degree = torch.zeros((num_nodes)).scatter_add_(0, source_index, torch.ones(source_index.shape)).to(torch.long)
#     in_degree = torch.zeros((num_nodes)).scatter_add_(0, target_index, torch.ones(target_index.shape)).to(torch.long)

#     return in_degree, out_degree

# def convert_edge_index_to_adjacency_list(edge_index, num_nodes, edge_weights = None):
#     adjacency_list = [[] for _ in range(num_nodes)]
#     source_vertices, target_vertices = edge_index   
    
#     if edge_weights is None:
#         edge_weights = torch.ones(edge_index.shape[-1])
    
#     for edge_id, (source_vertex, target_vertex, edge_weight) in enumerate(zip(source_vertices, target_vertices, edge_weights)):
#         source_vertex, target_vertex, edge_weight = source_vertex.item(), target_vertex.item(), edge_weight.item()

#         adjacency_list[source_vertex].append((target_vertex, edge_weight, edge_id))

#     return adjacency_list

# def trace_shortest_path(root_node, current_node, adjacency_list, distances, max_shortest_len):
#     if distances[current_node] == -1 or root_node == current_node:
#         return [-1] * max_shortest_len

#     trace, shortest_path = [current_node], []
#     while trace[-1] != root_node:
#         u = trace[-1]
        
#         for v, edge_len, edge_id in adjacency_list[u]:
#             if distances[u] == distances[v] + edge_len:
#                 trace.append(v)
#                 shortest_path.append(edge_id)
#                 break

#     shortest_path += [-1] * (max_shortest_len - len(shortest_path))

#     return shortest_path

# def find_shortest_path(root_node, num_nodes, adjacency_list, return_paths: bool = True):
#     # compute the shortest distance from root node to other nodes in the graph
#     # return the paths if needed

#     distances = [-1] * num_nodes
#     distances[root_node] = 0
#     shortest_distances = [(distances[root_node], root_node)]

#     while shortest_distances:
#         _, u = heapq.heappop(shortest_distances)

#         for v, edge_distance, _ in adjacency_list[u]:
#             if distances[v] == -1 or distances[u] + edge_distance < distances[v]:
#                 distances[v] = distances[u] + edge_distance
#                 heapq.heappush(shortest_distances, (distances[v], v))

#     if return_paths:
#         shortest_paths = []
#         for node_id in range(num_nodes):
#             shortest_path = trace_shortest_path(root_node, node_id, adjacency_list, distances, 5)
#             shortest_paths.append(shortest_path)

#         return distances, shortest_paths

#     return distances

# def shortest_path(num_nodes, edge_index, return_paths: bool = True):
#     adjacency_list = convert_edge_index_to_adjacency_list(edge_index, num_nodes)
    
#     # 2d array shortest_distances[i][j]: shortest distance from node i to node j, -1 , shape [N, N]
#     # if the path from i to j doesnt exist
#     shortest_distances = torch.from_numpy(np.array([find_shortest_path(root_node, num_nodes, adjacency_list, False) 
#                                         for root_node in range(num_nodes)])).to(torch.int64)
#     max_shortest_len = torch.max(shortest_distances)

#     # trace shortest path, shape [N, N, max_shortest_len]
#     # shortest_paths[i][j] = [e_1, e_2, ... ,e_n], the shortest path from node i to node j, e_i is the i_th edge's index
#     # padded with -1
#     if return_paths:
#         shortest_paths = np.array([[trace_shortest_path(root_node, node_id, adjacency_list, shortest_distances[root_node], max_shortest_len)
#                                 for node_id in range(num_nodes)] for root_node in range(num_nodes)])
#         shortest_paths = torch.from_numpy(shortest_paths).permute(1, 0, 2)

#         return shortest_distances, shortest_paths

#     return shortest_distances

def gaussian_kernel(x, mean, std):
    pi = torch.pi
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) * (-1 / (((2 * pi) ** 0.5) * std))

# z_in_degree + z_out_degree
class CentralityEncoding(nn.Module):
    def __init__(self, embedding_dim, max_degree, undirected_graph: bool) -> None:
        super(CentralityEncoding, self).__init__()
        
        if undirected_graph:
            self.degree_embedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)        
        else:
            self.in_degree_emdedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)
            self.out_degree_embedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)

        self.max_degree = max_degree
        self.undirected_graph = self.undirected_graph
 
    def forward(self, in_degree, out_degree):
        # in_degree, out_degree shape: [N]
        in_degree = torch.clamp(in_degree, min = 0, max = self.max_degree)
        out_degree = torch.clamp(out_degree, min = 0, max = self.max_degree)

        # shape: [N, embedding_dim]
        if self.undirected_graph:
            # unidirected_graph -> in_degree == out_degree 
            return self.degree_embedding(in_degree)
        else:
            return self.in_degree_emdedding(in_degree) + self.out_degree_embedding(out_degree)

class SpatialEncoding(nn.Module):
    def __init__(self, max_path_len, num_heads):
        super(SpatialEncoding, self).__init__()
        self.bias = nn.Embedding((max_path_len + 1) * num_heads, 1)
        self.max_path_len = max_path_len
        self.num_heads = num_heads
        
    def forward(self, shortest_distances, max_shortest_path_len):
        max_path_len = min(self.max_path_len, max_shortest_path_len)
        shortest_distances = torch.clamp(shortest_distances, min = 0, max = max_path_len)
        indices = [i for i in range((self.max_path_len + 1) * self.num_heads)]
        # [len, num_heads, 1] -> [len, 1, num_heads]
        spatial_encoding = self.bias(torch.tensor(indices).reshape(self.max_path_len + 1, self.num_heads)).permute(0, 2, 1)
        # shape [N, N, 1, num_heads] -> [N, N, num_heads]
        spatial_encoding = spatial_encoding[shortest_distances].squeeze(2)

        return spatial_encoding

class EdgeEncoding(nn.Module):
    def __init__(self, max_path_len, edge_feat_dim, num_heads):
        super(EdgeEncoding, self).__init__()
        self.max_path_len = max_path_len
        self.edge_feat_dim = edge_feat_dim
        self.num_heads = num_heads
        self.weight_embedding = nn.Embedding((max_path_len + 1) * num_heads, edge_feat_dim)

    def forward(self, shortest_paths, edge_feat, max_shortest_path_len):
        max_path_len = min(self.max_path_len, max_shortest_path_len)
        # shape [n, n, path_len]
        tmp_shortest_paths = shortest_paths[:, :, : max_path_len]
        # shape [n, n, path_len, edge_feat_dim]
        padded_edge_feat = torch.cat((edge_feat, torch.zeros(1, edge_feat.shape[-1])), dim = 0)
        edge_embedding = padded_edge_feat[tmp_shortest_paths]

        # reshape weight_embedding to [path_len, num_heads, edge_feat_dim] 
        # -> permute to get final shape [path_len, edge_feat_dim, num_heads]
        indices = [i for i in range(max_path_len * self.num_heads)]
        weight_embedding = self.weight_embedding(torch.tensor(indices).reshape(max_path_len, self.num_heads)).permute(0, 2, 1)
        
        # shape [N, N, path_len, edge_feat_dim], [path_len, edge_feat_dim, num_heads] -> [N, N, num_heads]
        edge_encoding = torch.einsum('xyld,ldh->xyh', edge_embedding.to(weight_embedding.dtype), weight_embedding)

        return edge_encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads, num_gaussian_heads, dropout, use_linear_bias = True):
        super(MultiHeadAttention, self).__init__()
        assert in_dim % num_heads == 0

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.num_gaussian_heads = num_gaussian_heads

        self.linear_Q = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_K = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_V = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_out = nn.Linear(in_dim, in_dim, bias = use_linear_bias)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        return

    def scaled_dot_attention(self, Q, K):
        attention_score = torch.matmul(Q, K.permute(1, 0))
        attention_score /= np.sqrt(self.head_dim)

        return attention_score

    def forward(self, node_feat, edge_index, distances, num_edge_types, max_path_len, edge_type, edge_feat = None, attention_mask = None):
        Q = self.linear_Q(node_feat)
        K = self.linear_K(node_feat)
        V = self.linear_V(node_feat)

        # [N, NH, head_dim]
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        # Q shape [n, num_heads, head_dim] -> [num_heads, N, head_dim]
        Q = Q.permute(1, 0, 2)
        # K shape [n, num_heads, head_dim] -> [num_heads, head_dim, N]
        K = K.permute(1, 2, 0)
        
        # matmul Q K -> [num_heads, n, n]
        attention_score = torch.matmul(Q, K)
        attention_score /= np.sqrt(self.head_dim)
        # permute attention_score shape -> [n, n, num_heads]
        attention_score = attention_score.permute(1, 2, 0)
        
        shortest_distances, shortest_paths = shortest_path(node_feat.shape[0], edge_index, return_paths = True)
        max_shortest_path_len = torch.max(shortest_distances)
        
        spatial_encoder = SpatialEncoding(max_path_len, self.num_heads)
        spatial_encoding = spatial_encoder(shortest_distances, max_shortest_path_len)

        attention_score += spatial_encoding
        if edge_feat is not None:
            edge_encoder = EdgeEncoding(max_path_len, edge_feat.shape[-1], self.num_heads)
            edge_encoding = edge_encoder(shortest_paths, edge_feat, max_shortest_path_len)

            attention_score += edge_encoding

        distance_encoder = DistanceEncoding(self.num_heads, self.num_gaussian_heads, num_edge_types)
        distance_encoding = distance_encoder(distances, edge_type)

        # attention_mask shape [n, n]
        if attention_mask is None:
            attention_mask = shortest_distances <= 0
        else:
            attention_mask = attention_mask.to(torch.bool)
            
        attention_score[attention_mask.to(torch.bool)] = float("-inf")

        # shape [n, n, num_heads]
        normalized_attention_score = nn.functional.softmax(attention_score, dim = -1)
        normalized_attention_score = self.dropout(normalized_attention_score)

        # normalized_attention_score, v -> out_node_feat
        # [n, n, num_heads], [n, num_heads, head_dim] -> [num_heads, n, n], [num_heads, n, head_dim]
        normalized_attention_score = normalized_attention_score.permute(2, 0, 1)
        V = V.permute(1, 0, 2)
        # -> [num_heads, n, head_dim] -> permute to [n, num_heads, head_dim] -> reshape to [n, num_heads * in_dim]
        out_node_feat = torch.matmul(normalized_attention_score, V)
        out_node_feat = out_node_feat.permute(1, 0, 2).view(-1, self.num_heads * self.head_dim)
        out_node_feat = self.linear_out(out_node_feat)

        return out_node_feat

class TransformerMLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout):
        super(TransformerMLayer, self).__init__()
        assert in_dim % num_heads == 0
        self.multihead_attention_layer_norm = nn.LayerNorm(in_dim)
        #def __init__(self, in_dim, num_heads, dropout, use_linear_bias = True):
        self.MultiHeadAttention_layer = MultiHeadAttention(in_dim, num_heads, dropout, True)
        
        self.ffn_layer_norm = nn.LayerNorm(in_dim)
        self.FNN_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout),)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, edge_index, max_path_len, edge_feat = None, attention_mask = None):
        # 1st Residual Connection: x = MHA(LN(x)) + x
        x_init = node_feat
        x = self.multihead_attention_layer_norm(node_feat)
        #def forward(self, node_feat, edge_index, max_path_len, edge_feat = None, attention_mask = None):        
        x = self.MultiHeadAttention_layer(x, edge_index, max_path_len, edge_feat, attention_mask)
        x = self.dropout(x)
        x = x + x_init

        # 2nd Residual Connection: x = FFN(LN(x)) + x
        x_init = x
        x = self.ffn_layer_norm(x)
        x = self.FNN_layer(x)
        x = x + x_init

        return x
    
class DistanceEncoding(nn.Module):
    def __init__(self, num_heads, K, num_edge_types, hidden_dim = None):
        super(DistanceEncoding, self).__init__()
        self.num_heads = num_heads
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.gammas = nn.Embedding(num_edge_types, 1, padding_idx = 0)
        self.betas = nn.Embedding(num_edge_types, 1, padding_idx = 0)

        if hidden_dim is None:
            hidden_dim = K
        self.linear_1 = nn.Linear(K, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_heads)

    def reset_parameters():
        return        

    def forward(self, distances, edge_type):
        num_nodes = distances.shape[0]
        # distances [n, n], edge_type[n, n]
        gammas = self.gammas(edge_type).squeeze(-1)
        betas = self.betas(edge_type).squeeze(-1)
        # shape [n, n]
        x = gammas * distances - betas
        # -> shape [n, n, k]
        x = torch.broadcast_to(x.unsqueeze(-1), (num_nodes, num_nodes, self.K))
        
        # shapae [n, n, k]
        means = torch.broadcast_to(self.means(torch.tensor([0])), (num_nodes, num_nodes, self.K))
        stds = torch.broadcast_to(self.stds(torch.tensor([0])), (num_nodes, num_nodes, self.K))

        distance_encoding = gaussian_kernel(x, means, stds)
        distance_encoding_without_nonlinear = distance_encoding

        # [n, n, k] -> [n, n, hidden_dim]
        distance_encoding = self.linear_1(distance_encoding)
        distance_encoding = nn.functional.gelu(distance_encoding)
        # [n, n, hidden_dim] -> [n, n, num_heads]
        distance_encoding = self.linear_2(distance_encoding)

        return distance_encoding

class StructuralEncoding(nn.Module):
    def __init__(self, in_dim, embedding_dim, K):
        super(StructuralEncoding, self).__init__()
        self.linear = nn.Linear(K, embedding_dim)
        
    def forward(self, distance_encoding_without_nonlinear):
        # [n, n, k] -> [n, n, d]
        distance_encoding_without_nonlinear = self.linear(distance_encoding_without_nonlinear)
        structural_encoding = torch.sum(distance_encoding_without_nonlinear, dim = 1)
        return structural_encoding