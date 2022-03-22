import numpy as np
import pandas as pd
from torch.utils import data


class GetData(data.Dataset):
    '''
    raw_data: [time_items_num, nodes, feature_num]
    seq_len: 
    pre_len:
    '''

    def __init__(self, src_data, trg_data, seq_len, pre_len, norm_type, attr:list=None):
        super().__init__()
        self.DataAttr = {'mean':[], 'std':[],'max':[],'min':[],'med':[]}
        src, trg = self.Norm(src_data, trg_data, norm_type, attr)
        self.seq_len = seq_len
        self.pre_len = pre_len

        srcx, srcy, trgx, trgy = [], [], [], []
        for i in range(len(src) - seq_len - pre_len):
            time_perio_label = np.asarray([t%288 for t in range(i, i+seq_len+pre_len)]).reshape(-1,1)
            a = np.concatenate((time_perio_label, src[i: i + seq_len + pre_len]),1) 
            srcx.append(a[0:seq_len])
            srcy.append(a[seq_len:seq_len+pre_len])
            b = np.concatenate((time_perio_label, trg[i: i + seq_len + pre_len]),1) 
            trgx.append(b[0:seq_len])
            trgy.append(b[seq_len:seq_len+pre_len])


        # In val : (34257, 12, 1401)
        self.srcx, self.trgx = np.asarray(srcx, dtype=np.float32), np.asarray(trgx, dtype=np.float32)
        self.srcy, self.trgy = np.asarray(srcy, dtype=np.float32), np.asarray(trgy, dtype=np.float32)
        

    def Norm(self, src, trg, norm, attr:list=None):
        if not attr[0]:
            src_attr = {'mean':np.mean(src), 'std':np.std(src),\
                        'max':np.max(src),'min':np.min(src),'med':np.median(src)}
        else:
            src_attr = attr[0]

        if not attr[1]:
            trg_attr = {'mean':np.mean(trg), 'std':np.std(trg),\
                        'max':np.max(trg),'min':np.min(trg),'med':np.median(trg)}
        else:
            trg_attr = attr[1]
        
        for key in self.DataAttr.keys():
            self.DataAttr[key].append(src_attr[key])
            self.DataAttr[key].append(trg_attr[key])
        # norm
        if norm == 'zscore':
            src = (src - self.DataAttr['mean'][0]) / self.DataAttr['std'][0]
            trg = (trg - self.DataAttr['mean'][1]) / self.DataAttr['std'][1]
        
        return src, trg

    def inverse(self, data, norm_type='zscore'):
        if norm_type == 'zscore':
            return data * self.DataAttr['std'][0] + self.DataAttr['mean'][0]
        
    def __len__(self):
        return len(self.srcx)

    def __getitem__(self, index):

        X = [self.srcx[index], self.trgx[index]]
        Y = [self.srcy[index], self.trgy[index]]
        return X, Y
