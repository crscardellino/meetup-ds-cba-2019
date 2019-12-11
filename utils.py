# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np

from scipy import sparse as sps
from tensorflow.keras.utils import to_categorical

np.random.seed(42)


def create_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_zachary_karate_club_data():
    """Function to load the Zachary's karate club dataset"""
    label_encoder = {
        "Mr. Hi": 0,
        "Officer": 1
    }
    G = nx.karate_club_graph()
    data = np.eye(len(G)).astype(np.float32)
    labels = np.array([
        label_encoder[G.nodes[n]["club"]] for n in sorted(G)
    ]).astype(np.float32)
    target = to_categorical(labels)
    adjacency = nx.adjacency_matrix(G).astype(np.float32)
    return G, data, target, adjacency


def plot_graph(G, labels, ax):
    """Helper function to plot a graph colored by the node labels"""
    nodes_colors = ["dodgerblue" if t == 0 else "tomato" for t in labels]

    # Mr. Hi and John A are highlighted
    nodes_colors[0] = "navy"
    nodes_colors[-1] = "darkred"

    pos = nx.spring_layout(G, iterations=75, seed=42)

    nx.draw_networkx_nodes(G, pos,
                           node_color=nodes_colors,
                           node_size=150,
                           ax=ax)
    nx.draw_networkx_edges(G, pos,
                           width=1,
                           ax=ax)


def sparse_to_tuple(spmx):
    """Convert sparse matrix to tuple representation."""

    if not sps.isspmatrix_coo(spmx):
        spmx = spmx.tocoo()

    indices = np.vstack((spmx.row, spmx.col)).transpose()
    values = spmx.data
    shape = spmx.shape
    return indices, values, shape


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sps.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sps.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model"""
    return normalize_adj(adj + sps.eye(adj.shape[0])).astype(np.float32)
