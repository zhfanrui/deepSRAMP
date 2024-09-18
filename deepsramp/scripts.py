import os
from . import *










def preprocess_data(args):
    print('preprocess')
    gtf = read_gtf(args.gtf)
    seqs = read_seq(args.fasta)
    if args.mode == 'mature':
        sdf = get_mature_splice(gtf, seqs, grp=['id', 'trans']) # TODO id/gene
        ndf = get_mature_cds(gtf, sdf, grp=['id', 'trans'])
    else:
        sdf = get_full_splice(gtf, seqs, grp=['id', 'trans'])
        ndf = get_full_cds(gtf, sdf, grp=['id', 'trans'])
    bed_names = os.listdir(args.m6a_bed_dir)
    bed_files = [os.path.join(args.m6a_bed_dir, i) for i in bed_names]
    res = read_m6a_beds(files, names, ndf)
    res = get_m6a_neg(res)
    res = res.dropna(subset=['m6a_pos'])
    utils.save(res, args.out)

def train_sramp(args):
    pass

def predict_sramp(args):
    pass

def preprocess_nano_data(args):
    split_drach(args.file)
    df, means, stds = read_merge_events(args.file, min_reads=1)
    df = df.dropna().groupby(level=[0, 1]).agg(list)
    # df = df[df['60'].apply(len) >= 20]
    save([df, means, stds], args.out)

def train_nano_sramp(args):
    pass

def predict_nano_sramp(args):
    pass

