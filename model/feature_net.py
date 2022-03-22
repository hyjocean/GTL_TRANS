from turtle import forward
import numpy as np
import torch 
import torch.nn as nn
from model.gru import GRU

class FeatureNet(nn.Module):
    def __init__(self, 
                adjs,
                hidden_dim: int,
                pre_len: int,
                bias: float=0.0):
        super().__init__()
        # self.input_dim = input_dim
        src_input_dim, trg_input_dim = adjs[0].shape[0], adjs[1].shape[0]
        self.hidden_dim = hidden_dim
        self.pre_len = pre_len
        self.bias_init = bias
        self.Src_gru = GRU(input_dim=src_input_dim, hidden_dim=hidden_dim)
        self.Trg_gru = GRU(input_dim=trg_input_dim, hidden_dim=hidden_dim)
        self.Batch_Norm_Layers = nn.BatchNorm1d(hidden_dim)
        self.trans_matrix_P = nn.Parameter(torch.FloatTensor(src_input_dim, trg_input_dim))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))

        self.linear = nn.Linear(hidden_dim, pre_len)
        self._init_para_()

    def _init_para_(self):
        nn.init.xavier_uniform_(self.trans_matrix_P)
        nn.init.constant_(self.bias, self.bias_init)

    def forward(self, src_x, trg_x):
        '''
        inp:
            x [SrcX, TrgX]      SrcX [b, seq_len, src_node+1] TrgX [b, seq_len, trg_node+1]
        '''
        # if self.training:
        batch_size, seq_len, src_num_nodes = src_x.size() 
        trg_num_nodes = trg_x.size(2) - 1
        hidden_src_fea = self.Src_gru(src_x[:,:,1:]) # [b, src_num_nodes, hidden_size]
        hidden_src_fea = self.Batch_Norm_Layers(hidden_src_fea.permute(0,2,1)).permute(0,2,1)
        hidden_trg_fea = self.Trg_gru(trg_x[:,:,1:]) # [b, trg_num_nodes, hidden_size]
        hidden_trg_fea = self.Batch_Norm_Layers(hidden_trg_fea.permute(0,2,1)).permute(0,2,1)
        hidden_src_fake_trg = torch.einsum('bsh, st->bth', hidden_src_fea, self.trans_matrix_P) # [b, trg_nodes, hidden_size]
        # TODO: 可以考虑P为概率转移矩阵，或者是否能够考虑成门控转移矩阵
        hidden_src_fake_trg = hidden_src_fake_trg.reshape((-1, self.hidden_dim)) + self.bias # [b*trg_num_nodes, hidden_size]
        hidden_trg_fea = hidden_trg_fea.reshape((-1, self.hidden_dim)) # [b*trg_num_ndoes, hidden_size]
        pre_src_fake_trg = self.linear(hidden_src_fake_trg) # [b*num_nodes, pre_len]
        pre_trg = self.linear(hidden_trg_fea)
        pre_src_fake_trg = pre_src_fake_trg.reshape((batch_size, trg_num_nodes, -1)).permute(0,2,1) # [b, pre_len, num_nodes]
        pre_trg = pre_trg.reshape((batch_size, trg_num_nodes, -1)).permute(0,2,1)
        return [[hidden_src_fake_trg, hidden_trg_fea], [pre_src_fake_trg, pre_trg]]

        # else:
        #     batch_size, seq_len, src_num_nodes = src_x.size() 
        #     hidden_src_fea = self.Src_gru(src_x[:,:,1:])
        #     hidden_src_fake_trg = torch.einsum('bsh, st->bth', hidden_src_fea, self.trans_matrix_P)
        #     hidden_src_fake_trg = hidden_src_fake_trg.reshape((-1, self.hidden_dim)) + self.bias
        #     pre = self.linear(hidden_src_fake_trg)
        #     pre = pre.reshape((batch_size, -1, self.pre_len))
        #     return pre

# class decoder(nn.Module):
#     def __init__(self, seq_len, pre_len, adjs, **kwargs):
#         super().__init__()
#         feature_dim=kwargs['faeture_dim']
#         fix_nd=kwargs['fix_nd']
#         self.pre_len = pre_len
#         self.time_weight = nn.parameter(torch.FloatTensor(pre_len, seq_len))
#         self.feature = feature_net(adjs, feature_dim, fix_nd, **kwargs)
#         self.pre_net = GRU(fix_nd, feature_dim)

#     def forward(self, x):
        
#         src_feat, trg_feat, src_fake_trg, src_net =self.feature(x)
#         # [12,b,fix_nd,hidden_size]
#         trium_data = torch.einsum('ni,ijkm->njkm', self.time_weight, src_fake_trg)
#         # [pre_len, b, fix_nd, hidden_size]
#         outputs = []
#         for t in range(self.pre_len):
#             out, hidden_state = self.pre_net(trium_data[t], hidden_state)
#             outputs.append(out)
    
#         return src_fake_trg, trg_feat, src_net, torch.stack(outputs)
