"""
Partition Network for CVRP.

Wraps EGNN to provide a simple interface for partition prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .egnn import EGNN


class PartitionNet(nn.Module):
    """
    E(2)-Equivariant Partition Network for CVRP.

    Predicts a heatmap H[i,j] representing the probability of transitioning
    from node i to node j during route construction.
    """

    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 node_input_dim=2, edge_input_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        """
        Args:
            n_layers: Number of EGNN layers
            hidden_dim: Hidden dimension
            node_dim: Node embedding dimension
            edge_dim: Edge embedding dimension
            node_input_dim: Input node features (demand/capacity, distance_from_depot)
            edge_input_dim: Input edge features (distance, affinity)
        """
        super().__init__()

        self.egnn = EGNN(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            coord_dim=2,
            coord_update_alpha=coord_update_alpha,
            weight_temp=weight_temp
        )

    def forward(self, graph_data):
        """
        Forward pass.

        Args:
            graph_data: Dictionary or object with:
                - x: Node features (n_nodes, node_input_dim)
                - pos: Node coordinates (n_nodes, 2)
                - edge_index: (2, n_edges)
                - edge_attr: (n_edges, edge_input_dim)

        Returns:
            heatmap: (n_nodes, n_nodes) transition probability matrix
        """
        # Extract graph components
        if isinstance(graph_data, dict):
            x = graph_data['x']
            pos = graph_data['pos']
            edge_index = graph_data['edge_index']
            edge_attr = graph_data['edge_attr']
            n_nodes = graph_data.get('n_nodes', x.shape[0])
            k_sparse = graph_data.get('k_sparse', edge_index.shape[1] // n_nodes)
        else:
            # Assume it's an object with attributes
            x = graph_data.x
            pos = graph_data.pos
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
            n_nodes = x.shape[0]
            k_sparse = edge_index.shape[1] // n_nodes

        # Get heatmap from EGNN
        heatmap = self.egnn.get_heatmap(
            node_features=x,
            edge_features=edge_attr,
            coords=pos,
            edge_index=edge_index,
            n_nodes=n_nodes,
            k_sparse=k_sparse
        )

        return heatmap

    def get_edge_logits(self, graph_data):
        """
        Get raw edge logits (before softmax).

        Useful for debugging or custom loss functions.
        """
        if isinstance(graph_data, dict):
            x = graph_data['x']
            pos = graph_data['pos']
            edge_index = graph_data['edge_index']
            edge_attr = graph_data['edge_attr']
        else:
            x = graph_data.x
            pos = graph_data.pos
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr

        return self.egnn.forward(x, edge_attr, pos, edge_index)
