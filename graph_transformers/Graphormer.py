import sys
sys.path.append('./')
sys.path.append('./utils')
sys.path.append('./transforms')

import torch
import torch.nn as nn
import numpy as np
from transforms.shortest_path import shortest_path

# z_in_degree + z_out_degree
class CentralityEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_degree: int, undirected_graph: bool):
        r"""
        Centrality Encoding from the ``_paper
        
        .. math:: 
            h_i^{(0)} = x_i + z^{-}_{\text{deg}^-(v_i)} + z^{+}_{\text{deg}^+{v_i}}

        Args:
            embedding_dim (int): size of degree embedding vectors
            max_degree (int): the maximum degree in the graph
            undirected_graph (bool): True if the graph is undirected and False if not 
        """
        
        super(CentralityEncoding, self).__init__()
        
        if undirected_graph:
            self.degree_embedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)        
        else:
            self.in_degree_emdedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)
            self.out_degree_embedding = nn.Embedding(max_degree + 1, embedding_dim, padding_idx = 0)

        self.max_degree = max_degree
        self.undirected_graph = self.undirected_graph
 
    def forward(self, in_degree, out_degree):
        r"""
        Parameters:
            in_degree (torch.tensor): 
                in_degree of each node. Shape: [N]
            out_degree (torch.tensor): 
                out_degree of each_node. Shape: [N]

        Returns
            degree embedidng (torch.tensor). Shape: [N]
        """                 

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
        r"""
        Spatial Encoding from

        Args:
            max_path_len (int): the maximum length of an encoded path
            num_heads (int): number of attention heads
        """
        super(SpatialEncoding, self).__init__()
        self.bias = nn.Embedding((max_path_len + 1) * num_heads, 1)
        self.max_path_len = max_path_len
        self.num_heads = num_heads
        
    def forward(self, shortest_distances, max_shortest_path_len):
        r"""
        Parameters:
            shortest_distances (torch.tensor):
                shortest distances between each pair of nodes. Shape: [N, N]
            max_shortest_path_len (int):
                maximum shortest distance between a pair of node

        Returns:
            spatial_encoding (torch.tensor):
                embedding vectors for spatial relations in the graph. Shape [N, N, num_heads]
        """
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
        r"""
        Edge Encoding from

        Args:
            max_path_len (int): maximum length of an encoded path
            edge_feat_dim (int): size of each edge's feature vector
            num_heads (int): number of attention heads
        """
        super(EdgeEncoding, self).__init__()
        self.max_path_len = max_path_len
        self.edge_feat_dim = edge_feat_dim
        self.num_heads = num_heads
        self.weight_embedding = nn.Embedding((max_path_len + 1) * num_heads, edge_feat_dim)

    def forward(self, shortest_paths, edge_feat, max_shortest_path_len):
        r"""
        Parameters:
            shortest_paths (torch.tensor):
                edge indices of the shortest path between each pair of nodes. Shape [N, N, max_shortest_path_len]
            edge_feat (torch.tensor):
                feature vector of edge. Shape [E, edge_feat_dim]
            max_shortest_path_len (int):
                maximum shortest distance between a pair of node

        Returns:
            edge_encoding (torch.tensor):
                embeddings vector for edges. Shape [N, N, num_heads]
        """
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
    def __init__(self, in_dim, num_heads, dropout, use_linear_bias = True):
        r"""
        MultiHeadAttention

        Args: 
            in_dim (int): size of nodes' input feature vector
            num_heads (int): number of attention heads
            dropout (float): dropout probability
            use_linear_bias (bool): True if bias is used in linear projections layers, False if not
        """
        super(MultiHeadAttention, self).__init__()
        assert in_dim % num_heads == 0

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.linear_Q = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_K = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_V = nn.Linear(in_dim, in_dim, bias = use_linear_bias)
        self.linear_out = nn.Linear(in_dim, in_dim, bias = use_linear_bias)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_Q.weight, gain = 2 ** (-0.5))
        nn.init.xavier_uniform_(self.linear_K.weight, gain = 2 ** (-0.5))
        nn.init.xavier_uniform_(self.linear_V.weight, gain = 2 ** (-0.5))
        
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_out.bias is not None:
            nn.init.constant_(self.linear_out.bias, 0.0)

    def forward(self, node_feat, edge_index, max_path_len, edge_feat = None, attention_mask = None):
        r"""
        Parameters:
            node_feat (torch.tensor):
                nodes' input feature vector. Shape [N, in_dim]
            edge_index (torch.tensor):
                Shape [2, E]
            max_path_len (int):
                maximum length of an encoded path
            edge_feat (None or torch.tensor):
                edges' input feature vector. Shape [E, edge_feat_dim]
            attention_mask (None or torch.tensor):
                invalid position for attention computation are assigned with non-zero values.
                Shape [N, N]

        Returns:
            out_node_feat (torch.tensor):
                nodes' output tensor vector. Shape [N, in_dim]
        """
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

        # attention_mask shape [n, n]
        if attention_mask is None:
            attention_mask = shortest_distances < 0
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
        # -> [num_heads, n, head_dim] -> permute to [n, num_heads, head_dim] -> reshape to [n, num_heads * head_dim]
        out_node_feat = torch.matmul(normalized_attention_score, V)
        out_node_feat = out_node_feat.permute(1, 0, 2).view(-1, self.num_heads * self.head_dim)
        out_node_feat = self.linear_out(out_node_feat)

        return out_node_feat

class GraphormerLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout):
        r"""
        GraphormerLayer

        Args:
            in_dim (int): size of each node's feature vector
            hidden_dim (int): size of hidden dim for feed forward network
            num_heads (int): number of attention heads
            dropout (float): dropout probability
        """
        super(GraphormerLayer, self).__init__()
        assert in_dim % num_heads == 0

        self.multihead_attention_layer_norm = nn.LayerNorm(in_dim)
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
        r"""
        Parameters:
            node_feat (torch.tensor):
                nodes' input feature vector. Shape [N, in_dim]
            edge_index (torch.tensor):
                Shape [2, E]
            max_path_len (int):
                maximum lenght of an encoded path
            edge_feat (None or torch.tensor):
                edges' input feature vector. Shape [E, edge_feat_dim]
            attention_mask (None or torch.tensor):
                invalid positions for attention computation are assigned with non-zero values.
                Shape [N, N]
            
        Returns:
            x (torch.tensor):
                nodes' output feature vector
        """
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