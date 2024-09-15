import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

from .data import genseq, genemb
from .utils import *

mers5 = [f'{i}{j}AC{k}' for i in 'ATG' for j in 'AG' for k in 'ATC']
mers5 += ([f'{k}{i}{j}AC' for i in 'ATG' for j in 'AG'  for k in 'ATCG'] + [f'{i}AC{j}{k}' for i in "AG" for j in 'ATC' for k in 'ATCG'])
MERDICT = dict(zip(mers5, np.eye(len(mers5))))
BASEDICT = dict(zip('ATCG', np.eye(4)))
K = 20

def norm(df, means, stds):
    tails = set()
    for i in means:
        tail = i[1:]
        tails.add(tail)
        for j in means[i]:
            means[i][j] = np.array(means[i][j])
    for i in stds:
        for j in stds[i]:
            stds[i][j] = np.array(stds[i][j])
        
    for tail in tails:
        kk = f'2{tail}'
        df[kk] = df[kk].apply(lambda x: x[0])
        for k in [6, 7, 8]:
            k = f'{k}{tail}'
            df[k] = df.apply(lambda x: (x[k] - means[k][x[kk]]) / stds[k][x[kk]], axis=1)

    return df

# def norm(df, means, stds):
#     for i in means:
#         for j in means[i]:
#             means[i][j] = np.array(means[i][j])
#     for i in stds:
#         for j in stds[i]:
#             stds[i][j] = np.array(stds[i][j])
            
#     for k in [6, 7, 8]:
#         df[k] = df.apply(lambda x: (x[k] - means[x[2]][k]) / stds[x[2]][k], axis=1)

#     for k in ['6_x', '7_x', '8_x']:
#         kk = int(k[0])
#         df[k] = df.apply(lambda x: (x[k] - means[x.seq[x[1]-3:x[1]+2]][kk]) / stds[x.seq[x[1]-3:x[1]+2]][kk], axis=1)

#     for k in ['6_y', '7_y', '8_y']:
#         kk = int(k[0])
#         df[k] = df.apply(lambda x: (x[k] - means[x.seq[x[1]-2:x[1]+3]][kk]) / stds[x.seq[x[1]-2:x[1]+3]][kk], axis=1)
        
#     return df

class NanoDS(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        current_seq = torch.tensor(self.data[idx][0], dtype=torch.float)
        current_pre = torch.tensor(self.data[idx][1], dtype=torch.float)
        current_data = torch.tensor(self.data[idx][2], dtype=torch.float)
        current_target = torch.tensor(self.target[idx], dtype=torch.float)
        
        return (current_seq, current_pre, current_data), current_target

def nanodf2ds(df, kodf=pd.DataFrame(columns=['trans', 'pos']), downsample=False, oversample=False):
    ndf = df[['trans', 'pos']].merge(kodf, on=['trans', 'pos'], how='left')
    xt = []
    yt = []
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    ratio = len(df_pos) / len(df_neg)
    np.random.seed(42)
    for trans, i in tqdm(df.reset_index().groupby(0)):
        pos = i[i['y'] == 1]
        neg = i[i['y'] == 0]

        try: ratio = len(pos) / len(neg) * 1
        except: ratio = 0

        for _, k in pos.iterrows():
            kok = ndf.loc[_]
            kok = kok if '20' in kok and kok['20'] == kok['20'] else None
            # print(_, kok)
            xt.append(np.concatenate([genx(k), genx(kok)], axis=0))
            yt.append(1)

        for _, k in neg.iterrows():
            if downsample and np.random.random() > ratio: continue
            # xt.append(genx(k))
            kok = ndf.loc[_]
            kok = kok if '20' in kok and kok['20'] == kok['20'] else None
            # print(_, kok)
            xt.append(np.concatenate([genx(k), genx(kok)], axis=0))
            yt.append(0)
            
        if oversample:
            for k in range(len(neg) - len(pos)):
                ttt = df_pos.sample(1).iloc[0]
                # xt.append(genx(ttt))
                kok = ndf.loc[ttt.name]
                kok = kok if '20' in kok and kok['20'] == kok['20'] else None
                xt.append(np.concatenate([genx(ttt), genx(kok)], axis=0))
                yt.append(1)
    return NanoDS(xt, yt)

def nanods2dl(ds, drop_last, num_workers):
    return DataLoader(ds, batch_size=128, drop_last=drop_last, num_workers=num_workers)

def nanodf2srampdf(traindf):
    predtraindf = traindf.groupby('trans').agg({
        'pos': set, 
        'seq': lambda x: x.tolist()[0], 
        'splice': lambda x: x.tolist()[0], 
        'cds': lambda x: x.tolist()[0], 
    })
    return predtraindf

def genx(x):
    np.random.seed(42)
    length = 3
    x = x.to_dict()
    
    # if x is None: return np.zeros((20, 2*length+1+8+9))
    # if x is None: return np.zeros((K, 66*length+9+11))
    
    # pretrainemb = np.array([x['pretrain']]*K).reshape((-1,len(x.pretrain)))
    
    if NANO_MODE == 'sramp':
        # seq = x['2-1'] + x['21'][-2:]
        # seq = genseq(seq, 3, length)
        seq = genseq(x['seq'], x['pos'], 66*length//2)[:-1]
        seqemb = np.array(seq)#.reshape((1, -1))#.repeat(K, axis=0)
        
        pretrainemb = genemb(x['seq'], x['pos'], 66*length//2, x)[:-1]
        # pretrainemb[:, [-1, -3, -4]] = 0
        # pretrainemb = pretrainemb#.reshape((1, -1))#.repeat(K, axis=0)
        
    else:
        seq = x['seq'][x['pos']-length:x['pos']+length+1]
        seqemb = []
        seqemb += MERDICT[seq[1:-1]].tolist()
        seqemb += MERDICT[seq[:-2]].tolist()
        seqemb += MERDICT[seq[2:]].tolist()
        seqemb = np.array(seqemb)#.reshape((1,-1))#.repeat(K, axis=0)
        pretrainemb = genemb(x['seq'], x['pos'], 0, x)
    
    # x = np.array(x[[f'{i}{j}' for i in [6, 7, 8] for j in [0, -1, 1, -2, 2][:3]]].to_numpy())
    x = np.array([x[f'{i}{j}'] for i in [6, 7, 8] for j in [0, -1, 1, -2, 2][:3]])
    
    idx = np.random.randint(x.shape[1], size=K)
    # idx = np.round((x.shape[1] - 1) * np.linspace(0, 1, K)).astype(int)
    x = x[:, idx]
    
    # x = x[:, np.argsort(x[0])]
    # x = x[:, :K]
    
    return seqemb, pretrainemb, x.T
    # return np.concatenate([seqemb, x.T], axis=1)
    # return np.concatenate([seqemb, x.T, pretrainemb], axis=1)


