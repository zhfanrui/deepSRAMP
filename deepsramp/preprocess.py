import pandas as pd
import numpy as np
# import regex as re
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import gzip
from . import utils

def read_seq(filename):
    seqs = {}
    name = ""
    seq = ""
    
    openfunc = gzip.open if filename.endswith('gz') else open
    with openfunc(filename, 'rb') as f:
        for i in tqdm(f):
            i = i.decode()
            if i.startswith(">"):
                seqs[name] = seq
                name = i[1:].split(' ')[0].strip().split('.')[0]
                seq = ""
            else:
                seq += i.strip().upper()
    seqs[name] = seq
    del seqs['']
    return seqs

def read_gtf(filename):
    gtf = pd.read_csv(filename, sep='\t', comment='#', header=None)
    gtf = gtf[gtf[2].isin(['exon', 'CDS'])]
    gtf[0] = gtf[0].astype('str').str.split('.', expand=True)[0]
    gtf = gtf[gtf[0].str.len() < 5]
    gtf[8] = gtf[8].apply(lambda x: dict(map(lambda y: y.split(' ', 1), x.split('; ')[:-1])))
    gtf['id'] = gtf[8].apply(lambda x: x['gene_id'].strip('"'))
    gtf['gene'] = gtf[8].apply(lambda x: x['gene_name'].strip('"') if 'gene_name' in x else None)
    gtf['trans'] = gtf[8].apply(lambda x: x['transcript_id'].strip('"'))
    gtf['biotype'] = gtf[8].apply(lambda x: x['transcript_biotype'].strip('"'))
    gtf['range'] = gtf.apply(lambda x: (x[3]-1, x[4]), axis=1)
    gtf['length'] = gtf[4] - gtf[3] + 1
    return gtf

def complement(s):
    for i, j in zip("AGCT", "1234"):
        s = s.replace(i, j)
    for i, j in zip("1234", "TCGA"):
        s = s.replace(i, j)
    return s

def get_mature_splice(gtf, seqs, biotype='protein_coding', grp=['gene', 'trans'], keepmax=False):
    sdf = gtf[((gtf['biotype'] == biotype) if biotype else True) & (gtf[2] == 'exon')].groupby(grp).agg({
        0: lambda x: x.tolist()[0],
        6: lambda x: x.tolist()[0],
        'range': lambda x: x.tolist(),
        'length': 'sum',
    })
    if keepmax:
        sdf = sdf.reset_index().groupby(grp[0]).apply(lambda x: x[x.length == x.length.max()].sample(1)).set_index(grp)
    sdf['splice'] = sdf.apply(lambda x: set([0] + np.cumsum([i[1] - i[0] for i in x.range]).tolist()), axis=1)
    sdf['min'] = sdf.apply(lambda x: min([i[0] for i in x.range]), axis=1)
    sdf['max'] = sdf.apply(lambda x: max([i[1] for i in x.range]), axis=1)
    sdf['seq'] = sdf.apply(lambda x: ''.join([seqs[x[0]][i:j] if x[6] == '+' else complement(seqs[x[0]][i:j][::-1]) for i, j in x.range]), axis=1)
    return sdf

def get_mature_cds(gtf, sdf, biotype='protein_coding', grp=['gene', 'trans']):
    ndf = gtf[((gtf['biotype'] == biotype) if biotype else True) & (gtf[2] == 'CDS')].groupby(grp).agg({
        'range': lambda x: x.tolist(),
    })
    ndf['range'] = ndf.apply(lambda x: (min([i[0] for i in x.range]), max([i[1] for i in x.range])), axis=1)
    ndf.columns = ['cds']
    ndf = ndf.merge(sdf, left_index=True, right_index=True, how='right')
    ndf['cds'] = ndf.apply(lambda x: x.cds if x.cds==x.cds else (0, 0), axis=1)
    ndf['cds'] = ndf.apply(lambda x: set((get_mature_pos(x.cds[0], x) + (1 if x[6] == '-' else 0), get_mature_pos(x.cds[1]-1, x) + (1 if x[6] == '+' else 0))), axis=1)
    return ndf
                              
def get_full_splice(gtf, sdf, seqs, grp=['gene', 'trans']):
    # ndf = gtf[(gtf['biotype'] == biotype) & (gtf[2] == 'transcript')].set_index(grp)[[3, 4]]
    # ndf = sdf.merge(ndf, left_index=True, right_index=True)
    # ndf['splice'] = ndf.apply(lambda x: [i[0] - x[3] + 1 if x[6] == '+' else x[4] - i[1] for i in x.range], axis=1)
    ndf = sdf.copy()
    ndf['splice'] = ndf.apply(lambda x: sum([[i, j] for i, j in x.range], []), axis=1)
    ndf['splice'] = ndf.apply(lambda x: set([i - x['min'] if x[6] == '+' else x['max'] - i for i in x.splice]), axis=1)
    ndf['seq'] = ndf.apply(lambda x: seqs[x[0]][x['min']:x['max']] if x[6] == '+' else complement(seqs[x[0]][x['min']:x['max']][::-1]), axis=1)
    ndf['length'] = ndf.seq.apply(len)
    return ndf

def get_full_cds(gtf, sdf, biotype='protein_coding', grp=['gene', 'trans']):
    ndf = gtf[((gtf['biotype'] == biotype) if biotype else True) & (gtf[2] == 'CDS')].groupby(grp).agg({
        'range': lambda x: x.tolist(),
    })
    ndf['range'] = ndf.apply(lambda x: (min([i[0] for i in x.range]), max([i[1] for i in x.range])), axis=1)
    ndf.columns = ['cds']
    ndf = ndf.merge(sdf, left_index=True, right_index=True)
    ndf['cds'] = ndf.apply(lambda x: set((get_full_pos(x.cds[0], x) + (1 if x[6] == '-' else 0), get_full_pos(x.cds[1]-1, x) + (1 if x[6] == '+' else 0))), axis=1)
    return ndf

def get_genome_pos(pos, sdf, mode='mature'):
    if mode == 'full': 
        if sdf[6] == '+':
            return pos + sdf['min']
        else:
            return sdf['max'] - pos - 1

    for idx, i in enumerate(sorted(sdf['splice'])):
        if i > pos: break
    n_exon = idx-1
    
    if sdf[6] == '+':
        return sdf['range'][n_exon][0] + pos - sorted(sdf['splice'])[n_exon]
    else:
        return sdf['range'][n_exon][1] - pos + sorted(sdf['splice'])[n_exon] - 1
    
    return -1

def get_mature_pos(k, sdf):
    for (i, j), d in zip(sdf['range'], sorted(sdf['splice'])):
        if i <= k < j:
            if sdf[6] == '+':
                return k-i+d
            else:
                return j-k+d-1
    return -1

def get_full_pos(k, sdf):
    if sdf['min'] <= k < sdf['max']:
        if sdf[6] == '+':
            return k-sdf['min']
        else:
            return sdf['max']-k-1
    return -1

def get_trans(ch, st, pos, sdf):
    # tsdf = sdf[(sdf[0] == ch) & (sdf[6] == st)]
    # tsdf = tsdf[(tsdf['min'] <= pos) & (tsdf['max'] > pos)]
    # tsdf = sdf.query(f'(0 == "{ch}") & (6 == "{st}")')
    # tsdf = tsdf.query('(min <= @pos) & (max > @pos)')
    
    tsdf = sdf[(sdf[0] == ch)]
    tsdf = tsdf[(tsdf[6] == st)]
    tsdf = tsdf[(tsdf['min'] <= pos)]
    tsdf = tsdf[(tsdf['max'] > pos)]
    return tsdf

DRACH_PAT = re.compile(r'(?=([ATG][AG]AC[ATC]))')
RRACH_PAT = re.compile(r'(?=([AG][AG]AC[ATC]))')

def read_m6a_bed(file, name, ndf, pat, get_pos):
    m6adf = pd.read_csv(file, sep='\t', header=None)
    m6adf = m6adf[m6adf[0].apply(len) < 6]
    m6adf[0] = m6adf[0].str.slice(3).replace({'M': 'MT'})
    tqdm.pandas(desc=name)
    m6adf['pos'] = m6adf.progress_apply(lambda x: [
        [
            gene, 
            trans, 
            (lambda y: y if y != -1 and len(re.findall(pat, i.seq[y-2:y+3])) else -1)(get_pos(x[1], i)),
            i.length # for max length
        ] for (gene, trans), i in get_trans(x[0], x[5], x[1], ndf).iterrows()
    ], axis=1)
    m6adf['pos'] = m6adf.pos.apply(lambda x: [i for i in x if i[2] != -1]) # drop -1
    m6adf['pos'] = m6adf.pos.apply(lambda x: [max(x, key=lambda y: y[3])[:3]] if x else []) # keep max length
    res = pd.DataFrame(sum(m6adf.pos, []), columns=ndf.index.names+[name])
    res = res[res[name] != -1]
    res = res.groupby(ndf.index.names).agg(lambda x: set(x))
    return res

def read_m6a_beds(files, names, ndf, pat=DRACH_PAT, get_pos=get_mature_pos):
    dfs = Parallel(n_jobs=20)([delayed(read_m6a_bed)(file, name, ndf, pat, get_pos) for file, name in zip(files, names)])
    df = pd.concat(dfs, axis=1)
    df['m6a_pos'] = df.apply(lambda x: set(sum([list(i) for i in x if i==i], [])), axis=1)
    df = pd.concat([ndf, df], axis=1)
    return df
    
def get_drach(seq, pat):
    drach = set()
    for j in re.finditer(pat, seq):
        pos = j.span()[0] + 2
        # pos = sum(j.span()) // 2
        drach.add(pos)
    return drach

def get_m6a_neg(sdf, pos_label='m6a_pos', pat=DRACH_PAT):
    sdf['m6a_neg'] = sdf.apply(lambda x: get_drach(x.seq, pat)-utils.notnaor(x[pos_label], set()), axis=1)
    return sdf

