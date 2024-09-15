import pickle
import contextlib
import joblib
from joblib import Parallel, delayed
import json
from tqdm import tqdm
import torch
import numpy as np

latent = 64
conv_len = 32
emb_len = 12
fea_len = 0
half_length = 500
length = 2 * half_length + 1
max_t = 32

# NANO_MODE = 'm6anet'
NANO_MODE = 'sramp'

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
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def isna(x):
    return x!=x

def isnotna(x):
    return x==x

def notnaor(x, y):
    if x==x: return x
    else: return y

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_model_config(file):
    with open(file) as f:
        obj = json.load(f)
    return obj