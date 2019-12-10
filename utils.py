# -*- coding: utf-8 -*-

import numpy as np

from scipy import sparse as sps


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


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
