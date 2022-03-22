import os
import numpy as np 
import pandas as pd 
import torch 
import scipy.sparse as sp 

from scipy.sparse import linalg
# from torch.nn.functional import normalize


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian

def dir_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        os.mknod(dirs+'/log.log')

def update(config, args):
    config['data']['seq_len'] = get_value(config['data']['seq_len'], args.seq_len)
    config['data']['pre_len'] = get_value(config['data']['pre_len'], args.pre_len)
    config['model']['seq_len'] = get_value(config['model']['seq_len'], args.seq_len)
    config['model']['pre_len'] = get_value(config['model']['pre_len'], args.pre_len)
    config['model']['gpu_id'] = get_value(config['model']['gpu_id'], args.gpu_id)
    config['model']['opt_lr'] = get_value(config['model']['opt_lr'], args.opt_lr)
    config['data']['norm_type'] = get_value(config['data']['norm_type'], args.norm_type)
    config['data']['data_file'] = get_value(config['data']['data_file'], args.data_file)
    config['data']['g_norm'] = get_value(config['data']['g_norm'], args.g_norm)
    return config

def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv

class Norm_data():
    def __init__(self, data, n_type='max-min', dim=None):
        '''
        inp:
            data: [b, seq_len, nodes, fea_num]
            dim: int
        '''
        self.data = data
        self.dim = dim
        self.type = n_type
        self.eps = 1e-8

    def get_attr(self):
        if self.dim is not None:
            std = torch.std(self.data, dim=self.dim, unbiased=True, keepdim=True)
            mean = torch.mean(self.data, dim=self.dim, keepdim=True)
            min_num = torch.min(self.data, dim=self.dim, keepdim=True)[0]
            max_num = torch.max(self.data, dim=self.dim, keepdim=True)[0]
            median = torch.median(self.data, dim=self.dim, keepdim=True)[0]
            self.attr_dic = {'mean':mean, 'std':std, 'max':max_num, 'min':min_num, 'median':median}
        else:
            std = np.std(self.data) # data:ndarray
            mean = np.mean(self.data)
            min_num = np.min(self.data)
            max_num = np.max(self.data)
            median = np.median(self.data)
            self.attr_dic = {'mean':mean, 'std':std, 'max':max_num, 'min':min_num, 'median':median}
        return self.attr_dic

    def norm(self):
        self.get_attr()
        if self.type == 'zscore':
            # z = (x - mean) / std
            return (self.data - self.attr_dic['mean']) / self.attr_dic['std']
        elif self.type == 'max-min':
            # z = (x - min) / (max - min)
            return (self.data - self.attr_dic['min']) / (self.attr_dic['max'] - self.attr_dic['min'] + self.eps)
        elif self.type == 'none':
            return self.data

    def inverse(self, data):
        '''
        inp: 
            data: [b, pre_len, nodes, fea_num]
        '''
        if self.type == 'zscore':
            # x = z*std + mean
            return (self.attr_dic['std'] * data) + self.attr_dic['mean']
        elif self.type == 'max-min':
            # x = z * (max - min) + min
            return data * (self.attr_dic['max'] - self.attr_dic['min'] + self.eps) + self.attr_dic['min']
        elif self.type == 'none':
            return data


## GRAPH wave
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
