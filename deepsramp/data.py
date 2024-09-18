import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import joblib, os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from . import utils


class MultiDS(Dataset):
    def __init__(self, data):
        self.data = data.reset_index(drop=True).to_dict()
        
    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        current_sample = torch.tensor(self.data['x1'][idx], dtype=torch.long)
        n = len(current_sample)
        current_sample = nn.functional.pad(current_sample, (0, 0, 0, utils.max_t-n), 'constant', 5)
        
        current_emb = torch.tensor(np.array(self.data['x2'][idx]), dtype=torch.float)
        current_emb = nn.functional.pad(current_emb, (0, 0, 0, 0, 0, utils.max_t-n), 'constant', -1)
        
        current_target = torch.tensor([(self.data['label'][idx][-1])], dtype=torch.float)
        
        return (current_sample, current_emb), current_target

def dffea_multi(df, pos_label='m6a_pos', neg_label='m6a_neg'):
    # partition = 10
    # perpar = ngenes // partition + 1
    # neg_df = Parallel(n_jobs=partition)(delayed(task_multi)(res[res.id.isin(genes[i*perpar:(i+1)*perpar])])  for i in range(partition))
    
    x = Parallel(n_jobs=10)(delayed(task_multi)(i.to_dict(), i.pos, pos_label, neg_label) for enst, i in tqdm(df.iterrows(), total=df.shape[0]))
    return list(zip(*x))

def task_multi(i, k, pos_label, neg_label):
    return (
        genseq(i['seq'], k, utils.half_length),
        genemb(i['seq'], k, utils.half_length, i, pos_label),
    )

def df2ds_multi(traindf, agg=True, return_grp=False):
    tdf = traindf[['trans', 'pos', 'label']].groupby(['label', 'trans']).agg(set)
    try:
        pos = tdf.loc[1]
        pos.columns = ['m6a_pos']
    except:
        pos = pd.DataFrame(columns=['m6a_pos'])
    try:
        neg = tdf.loc[0]
        neg.columns = ['m6a_neg']
    except:
        neg = pd.DataFrame(columns=['m6a_neg'])
        
    tdf = pos.merge(neg, left_index=True, right_index=True, how='outer')
    traindf = traindf.merge(tdf, left_on='trans', right_index=True)

    traindf['x1'], traindf['x2'] = dffea_multi(traindf)
    if agg:
        traindf = traindf[['x1', 'x2', 'grp', 'label']].groupby('grp').agg(list)
        if return_grp:
            return MultiDS(traindf), traindf.index
        else:
            return MultiDS(traindf1)
    else: 
        traindf['label'] = traindf['label'].apply(lambda x: [x])
        return DS(traindf[['x1', 'x2', 'label']].to_numpy().tolist())

class DS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = torch.tensor(self.data[idx][0], dtype=torch.long)
        current_emb = torch.tensor(self.data[idx][1], dtype=torch.float)
        current_target = torch.tensor(self.data[idx][-1], dtype=torch.float)

        return (current_sample, current_emb), current_target

def df2ds(df, downsample=False, neg_ratio=1, pos_label='m6a_pos', neg_label='m6a_neg'):
    tasklist = []
    for enst, i in tqdm(df.iterrows(), total=df.shape[0]):
        if utils.isnotna(i[pos_label]):
            if utils.isna(i[neg_label]) or len(i[neg_label]) == 0:
                i[neg_label] = set()
                ratio = 0
            else:
                ratio = len(i[pos_label]) / len(i[neg_label]) * neg_ratio

            for k in sorted(i[pos_label] | i[neg_label]):
                if downsample and k in i[neg_label] and np.random.random() > ratio: continue
                tasklist += [delayed(task)(i.to_dict(), k, pos_label, neg_label)]

    with utils.tqdm_joblib(tqdm(total=len(tasklist))) as pbar:
        x = Parallel(n_jobs=10)(tasklist)

    return DS(x)

def task(i, k, pos_label, neg_label):
    return (
        genseq(i['seq'], k, utils.half_length),
        genemb(i['seq'], k, utils.half_length, i, pos_label),
        genm6a(i['seq'], k, utils.half_length, i, pos_label, neg_label)[utils.half_length]
    )

def ds2dl(ds, drop_last=False, num_workers=0, shuffle=True, batch_size=256):
    return DataLoader(ds, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers)

def backseq(s):
    d = dict(zip(range(6), 'AUCGNN'))
    rs = ''
    for i in s.tolist():
        rs += d[i]
    return rs

def read_data(filename='all.pk'):
    with open(filename, 'rb') as f:
        pretrainds = pickle.load(f)
    return pretrainds

def splitdf(df, test_size=0.2, label='y'):
    p_train, p_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df[label])
    return p_train, p_test

def ssum(l, s=0):
    for i in tqdm(l):
        s += i
    return s

def gencl(k, i):
    r1, r2 = None, None
    for ds in i.index:
        if isinstance(ds, int) or '_' not in ds or ds in ['m6a_pos', 'm6a_neg']: continue
        j = ds.split('_')
        if r1 is None:
            r1 = np.zeros(int(j[2].split('/')[1]))
            r2 = np.zeros(int(j[3].split('/')[1]))
        if isinstance(i[ds], set) and k in i[ds]:
            r1[int(j[2].split('/')[0])] = 1
            r2[int(j[3].split('/')[0])] = 1
    if sum(r1) == 0:
        r1[:] = 1
        r2[:] = 1
    return np.concatenate([r1, r2])

def genseq(seq, k, w):
    s = ""
    if k - w < 0:
        s += "Z" * (w-k)
    s += seq[max(k-w, 0):k+w+1]
    if k + w + 1 > len(seq):
        s += "Z" * (k + w + 1 - len(seq))
    base = 'ATCGNZ'
    dic = dict(zip(base, range(len(base))))
    sl = [dic[i] if i in base else dic['N'] for i in s]
    # sl = []
    # for i in s:
    #     if i in base: sl.append(dic[i])
    #     else: sl.append(dic['N'])
    return sl

def keep_middle(arr):
    arr[:utils.half_length] = 0
    arr[utils.half_length+1:] = 0
    return arr

def genemb(seq, k, w, df, pos_label='m6a_pos'):
    fea = []
    length = len(seq)

    splices = sorted(df['splice'])
    n_exon = len(splices) - 1

    idx = np.arange(k-w, k+w+1)

    # position
    fea += [idx / length]

    mincds, maxcds = min(df['cds']), max(df['cds'])

    # cds
    fea += [(mincds <= idx) & (idx < maxcds)]

    # ssc
    fea += [((idx - mincds) / length)]
    fea += [((idx - maxcds) / length)]

    # splice
    fea += [np.isin(idx, splices)]

    # exon length
    tfea = [np.zeros(2*w+1) for i in range(4)]
    for j in range(n_exon):
        l, r = splices[j], splices[j+1]
        exon_len = (r - l) / length

        cond = (l <= idx) & (idx < r)

        tfea[0][cond] = exon_len
        tfea[1][cond] = j / n_exon
        tfea[2][cond] = (idx - l)[cond] / length
        tfea[3][cond] = (idx - r)[cond] / length

    fea += tfea

    # m6a
    if utils.isna(df[pos_label]): df[pos_label] = set()
    m6a = np.array([list(df[pos_label] | set([2147483647]))]) - idx.reshape((-1, 1))
    m6a[m6a<1] = 200
    # m6a = np.where(m6a < 1, length - idx.reshape((-1, 1)), m6a)
    # m6a[m6a>200] = 200
    m6a = keep_middle(m6a)
    fea += [m6a.min(axis=1)]

    m6a = idx.reshape((-1, 1)) - np.array([list(df[pos_label] | set([-2147483647]))])
    m6a[m6a<1] = 200
    # m6a = np.where(m6a < 1, idx.reshape((-1, 1)), m6a)
    # m6a[m6a>200] = 200
    m6a = keep_middle(m6a)
    fea += [m6a.min(axis=1)]

    m6a = np.isin(idx, list(df[pos_label] - set([k])))
    fea += [m6a]

#     m6a = np.array(df[f'{fro1}_exp']).repeat(idx.shape[0])
#     m6a = keep_middle(m6a)
#     fea += [m6a]

    fea = np.array(fea).T
    return fea

def genm6a(seq, k, w, df, pos_label='m6a_pos', neg_label='m6a_neg'):
    fea = []
    idx = np.arange(k-w, k+w+1)

    # fea += [np.isin(idx, list(df['splice']))]
    # fea += [np.isin(idx, list(df['cds']))]
    parr = np.isin(idx, list(df[pos_label]))
    narr = np.isin(idx, list(df[neg_label]))
    fea += [2*parr + narr - 1]
    fea = np.array(fea).T
    return fea

def genei(df, k):
    sp = sorted(df['splice'])
    sp1 = sp[::2]
    sp2 = sp[1::2]
    for i, j in zip(sp1, sp2):
        if i <= k < j:
            return 1
    return 0

def integrate_pred(pred, res):
    res['drach'] = res.apply(lambda x: sorted((x.m6a_pos if x.m6a_pos == x.m6a_pos else set()) | x.m6a_neg), axis=1)
    res['drach_len'] = res.apply(lambda x: len(x.drach), axis=1)
    res['drach_len'] = np.cumsum([0]+res.drach_len.tolist())[:-1]
    allsites = sum(res.apply(lambda x: [[x.name, i, pred[x.drach_len + idx]] for idx, i in enumerate(x.drach)], axis=1), [])
    allsites = pd.DataFrame(allsites, columns=[0, 1, 'pretrain'])
    return allsites
