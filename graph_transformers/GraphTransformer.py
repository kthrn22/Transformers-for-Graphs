import torch
import torch.nn as nn
import numpy as np

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_edge_feat = False, use_bias = False, use_layer_norm = False, use_batch_norm = True, residual_connection = True, dropout = 0.0):
        super(GraphTransformerLayer, self).__init__()
        assert out_dim % num_heads == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.linear_Q = nn.Linear(in_dim, out_dim, bias = use_bias)
        self.linear_K = nn.Linear(in_dim, out_dim, bias = use_bias)
        self.linear_V = nn.Linear(in_dim, out_dim, bias = use_bias)
        
        self.linear_O = nn.Linear(out_dim, out_dim)
        if use_edge_feat:
            self.linear_E = nn.Linear(in_dim, out_dim, bias = use_bias)
            self.linear_O_edge = nn.Linear(out_dim, out_dim)

        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)

            if use_edge_feat:
                self.layer_norm1_edge = nn.LayerNorm(out_dim)
                self.layer_norm2_edge = nn.LayerNorm(out_dim)
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

            if use_edge_feat:
                self.batch_norm1_edge = nn.BatchNorm1d(out_dim)
                self.batch_norm2_edge = nn.BatchNorm1d(out_dim)
        self.use_batch_norm = use_batch_norm


        self.FFN_Layer_1 = nn.Linear(out_dim, 2 * out_dim)
        self.FFN_Layer_2 = nn.Linear(2 * out_dim, out_dim)
        if self.use_edge_feat:
            self.FFN_Layer_1_edge = nn.Linear(out_dim, 2 * out_dim)
            self.FFN_Layer_2_edge = nn.Linear(2 * out_dim, out_dim)

        self.use_edge_feat = use_edge_feat

        if residual_connection:
            assert out_dim == in_dim
        self.residual_connection = residual_connection

        self.dropout = dropout

    def forward(self, node_feat, edge_index, edge_feat = None):
        assert edge_index.shape[0] == 2
        if self.use_edge_feat:
            assert edge_feat is not None

        ## node_feat.shape: [N, in_dim]; edge_index.shape: [2, num_edges]
        num_nodes, num_edges = node_feat.shape[0], edge_index.shape[-1]
        Q, K, V = self.linear_Q(node_feat), self.linear_K(node_feat), self.linear_V(node_feat)
        
        # shape = [num_nodes, num_heads, head_dim]
        # [N, NH, DIM]
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)
        E = None

        if self.use_edge_feat:
            # edge_feat: [E, in_dim]
            E = self.linear_E(edge_feat)
            E = E.view(-1, self.num_heads, self.head_dim)

        # [E, NH]
        normalized_attention_per_edge = self.normalize_attention_score(Q, K, edge_index, num_nodes, E)

        source_index, target_index = edge_index
        # [E] -> [E, NH, DIM]
        target_index = torch.broadcast_to(target_index.unsqueeze(-1).unsqueeze(-1), (num_edges, self.num_heads, self.head_dim))
        
        # [N, NH, DIM]
        out_node_feat = torch.zeros(num_nodes, self.num_heads, self.head_dim).to(node_feat.dtype)
        out_node_feat.scatter_add_(0, target_index, normalized_attention_per_edge.unsqueeze(-1) * V[source_index])
        out_node_feat = out_node_feat.view(-1, self.out_dim)
        out_node_feat = nn.functional.dropout(out_node_feat, self.dropout, training = self.training)
        out_node_feat = self.linear_O(out_node_feat)

        if self.use_edge_feat:
            attention_score_per_edge = self.attention_per_edges(Q, K, edge_index, E)
            attention_score_per_edge = torch.broadcast_to(attention_score_per_edge.unsqueeze(-1), (num_edges, self.num_heads, self.head_dim))
            out_edge_feat = attention_score_per_edge.view(-1, self.out_dim)
            out_edge_feat = nn.functional.dropout(out_edge_feat, self.dropout, training = self.training)
            out_edge_feat = self.linear_O_edge(attention_score_per_edge)

        # 1st Residual Connection + Norm
        x_1 = node_feat
        if self.residual_connection:
            out_node_feat = out_node_feat + x_1
    
        if self.use_layer_norm:
            out_node_feat = self.layer_norm1(out_node_feat)
        
        if self.use_batch_norm:
            out_node_feat = self.batch_norm1(out_node_feat)

        # For 2nd Residual Connection
        x_2 = out_node_feat
        # FFN 
        out_node_feat = self.FFN_Layer_1(out_node_feat) 
        out_node_feat = nn.functional.relu(out_node_feat)
        out_node_feat = self.FFN_Layer_2(out_node_feat)

        # 2nd Residual Connection + Norm
        if self.residual_connection:
            out_node_feat = out_node_feat + x_2

        if self.use_layer_norm:
            out_node_feat = self.layer_norm2(out_node_feat)

        if self.use_batch_norm:
            out_node_feat = self.batch_norm2(out_node_feat)

        if self.use_edge_feat:
            e_1 = edge_feat
            if self.residual_connection:
                out_edge_feat = out_edge_feat + e_1
        
            if self.use_layer_norm:
                out_edge_feat = self.layer_norm1_edge(out_edge_feat)
            
            if self.use_batch_norm:
                out_edge_feat = self.batch_norm1_edge(out_edge_feat)

            # For 2nd Residual Connection
            e_2 = out_edge_feat
            # FFN 
            out_edge_feat = self.FFN_Layer_1_edge(out_edge_feat) 
            out_edge_feat = nn.functional.relu(out_edge_feat)
            out_edge_feat = self.FFN_Layer_2_edge(out_edge_feat)

            # 2nd Residual Connection + Norm
            if self.residual_connection:
                out_edge_feat = out_edge_feat + e_2

            if self.use_layer_norm:
                out_edge_feat = self.layer_norm2_edge(out_edge_feat)

            if self.use_batch_norm:
                out_edge_feat = self.batch_norm2_edge(out_edge_feat)

            return out_node_feat, out_edge_feat

        return out_node_feat

    def attention_per_edges(self, Q, K, edge_index, E = None):
        source_index, target_index = edge_index
        if E is None:
            # source: K, target: Q
            # shape = [E, NH]
            attention_score_per_edge = torch.sum(Q[target_index] * K[source_index], dim = -1)
            attention_score_per_edge /= np.sqrt(self.head_dim)
            attention_score_per_edge = torch.exp(attention_score_per_edge)

            return attention_score_per_edge
        
        else:
            attention_score_per_edge = torch.sum(Q[target_index] * K[source_index] * E, dim = -1)
            attention_score_per_edge /= np.sqrt(self.head_dim)

            return attention_score_per_edge
        
    def aggregrate_attention_score(self, Q, K, edge_index, num_nodes, E = None):
        # shape = [E, NH]
        attention_score_per_edge = self.attention_per_edges(Q, K, edge_index)
        if E is not None:
            attention_score_per_edge = torch.exp(attention_score_per_edge)

        # [E] -> [E, NH]
        target_index = torch.broadcast_to(edge_index[1].view(-1, 1), (edge_index.shape[1], self.num_heads))
        # [N, NH]
        sum_attention_score = torch.zeros((num_nodes, self.num_heads)).to(attention_score_per_edge.dtype)
        sum_attention_score.scatter_add_(0, target_index, attention_score_per_edge)
        
        return sum_attention_score

    def normalize_attention_score(self, Q, K, edge_index, num_nodes, E = None):
        source_index, target_index = edge_index
        attention_score_per_edge = self.attention_per_edges(Q, K, edge_index, E) # [E, NH]
        sum_attention_score = self.aggregrate_attention_score(Q, K, edge_index, num_nodes, E) # [N, NH]

        # [E, NH]
        normalized_attention_per_edge = attention_score_per_edge / sum_attention_score[target_index]

        return normalized_attention_per_edge