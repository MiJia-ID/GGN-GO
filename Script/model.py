import numpy as np
import torch
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch.nn import functional as F
from pool import GraphMultisetTransformer
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool as gmp


class GVP_encoder(nn.Module):  # Define GVP_encoder class, inheriting from nn.Module
    """
    GVP-GNN for Model Quality Assessment as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    """

    def __init__(
        self,
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        seq_in=False,
        num_layers=3,
        drop_rate=0.1,
    ):
        # Initialization function to set the class attributes, including input dimensions, hidden dimensions, number of layers, and dropout rate
        super(
            GVP_encoder, self
        ).__init__()  # Call the initialization function of the parent class
        activations = (
            F.relu,
            None,
        )  # Set activation functions, the first is ReLU, the second is None

        self.seq_in = (
            seq_in  # Store the flag indicating whether sequence input is needed
        )
        if self.seq_in:  # If sequence input is needed
            self.W_s = nn.Embedding(
                21, 21
            )  # Define an Embedding layer to embed sequence indices into 21-dimensional vectors
            node_in_dim = (
                node_in_dim[0] + 21,
                node_in_dim[1],
            )  # Adjust the node input dimensions to include sequence features

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),  # Apply LayerNorm normalization to node inputs
            GVP(
                node_in_dim, node_h_dim, activations=(None, None), vector_gate=True
            ),  # Process nodes using the GVP layer
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),  # Apply LayerNorm normalization to edge inputs
            GVP(
                edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True
            ),  # Process edges using the GVP layer
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_h_dim,
                edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
            )
            for _ in range(num_layers)
        )  # Create multiple GVP convolution layers

        ns, _ = (
            node_h_dim  # Extract the scalar feature count from node hidden dimensions
        )
        self.W_out = nn.Sequential(
            LayerNorm(
                node_h_dim
            ),  # Apply LayerNorm normalization to node hidden features
            GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=True),
        )  # Output node features using the GVP layer

        self.dense = nn.Sequential(
            nn.Linear(ns, ns),
            nn.ReLU(inplace=True),  # Define a linear layer and ReLU activation
            nn.Dropout(p=drop_rate),  # Add a Dropout layer
            # nn.Linear(2*ns, 1)  # Commented out linear layer
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None, pertubed=False):
        """
        Forward pass function that processes input nodes, edges, and optional sequence features

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        """
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)  # Embed the sequence
            h_V = (
                torch.cat([h_V[0], seq], dim=-1),
                h_V[1],
            )  # Append the embedded sequence to the node features

        h_V = self.W_v(h_V)  # Process node features using the GVP layer
        h_E = self.W_e(h_E)  # Process edge features using the GVP layer

        for layer in self.layers:
            if pertubed:  # If perturbed is True
                h_V = layer(
                    h_V, edge_index, h_E, pertubed=True
                )  # Process node features with perturbation parameters
            else:
                h_V = layer(
                    h_V, edge_index, h_E
                )  # Process node features without perturbation

        out = self.W_out(h_V)  # Output node features using the GVP layer

        return self.dense(out)  # Return the output of the dense layer


class GGN_GO(nn.Module):  # Define GGN_GO class, inheriting from nn.Module
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        augment_eps,
        dropout,
        out_dim,
        pooling="MTP",
        pertub=False,
    ):
        # Initialization function to set the class attributes, including input dimensions, hidden dimensions, number of layers, perturbation, dropout rate, output dimensions, and pooling method
        super(
            GGN_GO, self
        ).__init__()  # Call the initialization function of the parent class

        self.pertub = pertub  # Store the flag indicating whether perturbation is needed
        self.GVP_encoder = GVP_encoder(
            node_in_dim=(input_dim, 3),
            node_h_dim=(hidden_dim, 16),
            edge_in_dim=(32, 1),
            edge_h_dim=(32, 1),
            seq_in=True,
            num_layers=num_layers,
            drop_rate=dropout,
        )
        # Initialize the GVP_encoder object, setting parameters for node and edge input and hidden dimensions

        self.pooling = pooling  # Store the pooling method

        if self.pooling == "MTP":  # If pooling method is "MTP"
            self.pool = GraphMultisetTransformer(
                hidden_dim,
                256,
                hidden_dim,
                None,
                10000,
                0.25,
                ["GMPool_G", "GMPool_G"],
                num_heads=8,
                layer_norm=True,
            )
            # Initialize GraphMultisetTransformer object for graph multiset pooling
        else:
            self.pool = gmp  # Otherwise, use global max pooling

        self.readout = nn.Sequential(
            # nn.Linear(128, 512),  # Commented out linear layer
            # nn.BatchNorm1d(512),  # Commented out batch normalization layer
            nn.Linear(
                hidden_dim, 1024
            ),  # Define a linear layer mapping hidden dimension to 1024
            nn.ReLU(),  # Use ReLU activation function
            nn.Dropout(0.2),  # Add a Dropout layer with a dropout rate of 0.2
            nn.Linear(
                1024, out_dim
            ),  # Define another linear layer mapping 1024 to output dimension
            nn.Sigmoid(),  # Use Sigmoid activation function
        )

        self.softmax = nn.Softmax(
            dim=-1
        )  # Define a Softmax layer for final classification

        # Initialize network weights
        for p in self.parameters():  # Iterate over all network parameters
            if p.dim() > 1:  # If the parameter has more than 1 dimension
                nn.init.xavier_uniform_(p)  # Use Xavier uniform initialization

    def forward(self, h_V, edge_index, h_E, seq, batch=None):
        # Forward pass function that processes input nodes, edges, and sequence features

        h_V_ = self.GVP_encoder(
            h_V, edge_index, h_E, seq
        )  # Process node features through GVP_encoder

        if self.pooling == "MTP":  # If pooling method is "MTP"
            x = self.pool(
                h_V_, batch, edge_index.long()
            )  # Use GraphMultisetTransformer for pooling
        else:
            x = self.pool(h_V_, batch)  # Otherwise, use global max pooling

        if self.pertub:
            h_V_pertub = self.GVP_encoder(
                h_V, edge_index, h_E, seq, pertubed=True
            )  # Process node features again with perturbed parameters

            if self.pooling == "MTP":  # Use GraphMultisetTransformer for pooling
                x_pertub = self.pool(h_V_pertub, batch, edge_index.long())
            else:
                x_pertub = self.pool(h_V_pertub, batch)  # Use global max pooling

            y_pred = self.readout(x)  # Get prediction results through the readout layer

            return y_pred, x, x_pertub  # Return prediction results and pooled outputs

        else:
            y_pred = self.readout(x)  # Get prediction results through the readout layer

            return y_pred  # Return prediction results
            # return y_pred,h_V_out  # Commented out return statement
