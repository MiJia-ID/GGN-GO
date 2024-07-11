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


class GVP_encoder(nn.Module):
    '''
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
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_encoder, self).__init__()
        activations = (F.relu, None)

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(21, 21)
            node_in_dim = (node_in_dim[0] + 21, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, activations=activations, vector_gate=True, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=True))

        self.dense = nn.Sequential(
            nn.Linear(ns, ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            # nn.Linear(2*ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None, pertubed=False):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            if pertubed:
                h_V = layer(h_V, edge_index, h_E, pertubed=True)
            else:
                h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)

        return self.dense(out)

class GLMSite(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, augment_eps, dropout, out_dim, pooling='MTP', pertub=False):
        super(GLMSite, self).__init__()

        self.pertub = pertub
        self.GVP_encoder = GVP_encoder(node_in_dim=(input_dim, 3), node_h_dim=(hidden_dim, 16), edge_in_dim=(32, 1),
                                       edge_h_dim=(32, 1), seq_in=True, num_layers=num_layers, drop_rate=dropout)

        self.pooling = pooling

        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(hidden_dim, 256, hidden_dim, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'],
                                                 num_heads=8, layer_norm=True)
        else:
            self.pool = gmp

        self.readout = nn.Sequential(
            # nn.Linear(128, 512),
            # nn.BatchNorm1d(512),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=-1)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, h_V, edge_index, h_E, seq, batch=None):

        h_V_ = self.GVP_encoder(h_V, edge_index, h_E, seq)  # [num_residue, hidden_dim]

        if self.pooling == "MTP":
            x = self.pool(h_V_, batch, edge_index.long())
        else:
            x = self.pool(h_V_, batch)

        if self.pertub:
            h_V_pertub = self.GVP_encoder(h_V, edge_index, h_E, seq, pertubed=True)

            if self.pooling == "MTP":
                x_pertub = self.pool(h_V_pertub, batch, edge_index.long())
            else:
                x_pertub = self.pool(h_V_pertub, batch)

            y_pred = self.readout(x)

            return y_pred, x, x_pertub

        else:
            y_pred = self.readout(x)

            return y_pred
            #return y_pred,h_V_out

