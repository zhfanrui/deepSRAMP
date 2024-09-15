import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchComple

def eventalign(fasta, bam, genome, out):
    s = f'''nanopolish eventalign \
    --threads 196 \
    --progress \
    --reads {fasta} \
    --bam {bam} \
    --genome {genome} \
    --signal-index \
    --scale-events > \
    awk '$3 ~ /[ATG][AG]AC[ATC]/ || $3 ~ /[ATCG][ATG][AG]AC/ || $3 ~ /[AG]AC[ATC][ATCG]/ || $3 ~ /[ATCG][ATCG][ATG][AG]A/ || $3 ~ /AC[ATC][ATCG][ATCG]/ {{print}}' \
    > {out}'''
    os.system(s)
    return

def split_drach(file):
    patterns = ['[ATG][AG]AC[ATC]', '[ATCG][ATG][AG]AC', '[AG]AC[ATC][ATCG]', '[ATCG][ATCG][ATG][AG]A', 'AC[ATC][ATCG][ATCG]']
    names = [0, -1, 1, -2, 2]
    
    for p, n, in tqdm(zip(patterns, names), total=5):
        os.system(f"awk '$3 ~ /{p}/ {{print}}' {file} > {file}{n}.txt")

        
# def task(name, group, func):
#     return func(group)

# def applyParallel(groups, func, l):
#     with tqdm_joblib(tqdm(total=l)) as pbar:
#         ret = Parallel(n_jobs=4)(delayed(task)(name, group, func) for name, group in groups)
#     return pd.concat(ret, axis=1).T

def read_events(file, offset=0):
    df = pd.read_csv(file, sep=' ', header=None)
    df.columns = [0, 1, 2, 3, 6, 7, 8, 13, 14]
    df[0] = df[0].apply(lambda x: x.split('.')[0] if not x.upper().startswith('AT') else x)
    df[1] += 2 - offset
    # df[8] = df[14] - df[13]
    df = df.iloc[:, :-2]
    df[6] = df[6] * df[8]
    df[7] = df[7] * df[7] * df[8]
    df = df.groupby([0, 1, 2, 3]).agg('sum')
    # df = applyParallel(df.groupby([0, 1, 2, 3]), np.sum)
    df[6] /= df[8]
    df[7] = np.sqrt(df[7]/df[8])
    df = df.reset_index().set_index([0, 1, 3])
    df.columns = [f'2{offset}', f'6{offset}', f'7{offset}', f'8{offset}']
    mean = df.groupby(f'2{offset}').agg('mean').to_dict()
    # mean = applyParallel(df.groupby(f'2{offset}'), np.mean).to_dict()
    std = df.groupby(f'2{offset}').agg('std').to_dict()
    # std = applyParallel(df.groupby(f'2{offset}'), np.std).to_dict()
    return df, mean, std

def read_merge_events(file, min_reads=1):
    names = [0, -1, 1, -2, 2]
    fdf = pd.DataFrame()
    means = {}
    stds = {}
    for i in tqdm(names[:3]):
        f = f'{file}{i}.txt'
        df, mean, std = read_events(f, offset=i)
        means = {**means, **mean}
        stds = {**stds, **std}
        if fdf.empty:
            fdf = df
            # return df
            # fdf = applyParallel(df.groupby(level=[0,1], sort=False), lambda x: x if x.shape[0] >= 1 else None, df.reset_index()[[0, 1]].value_counts().shape[0])
            # tqdm.pandas()
            # fdf = df.groupby(level=[0, 1]).progress_apply(lambda x: x if x.shape[0] >= min_reads else None)#.droplevel([0, 1])
        else: fdf = fdf.merge(df, left_index=True, right_index=True, how='left')
    return fdf, means, stds

