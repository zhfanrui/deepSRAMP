# from .utils import load, save, isna, isnaor
from . import utils
from .preprocess import *
from .model import *
from .train import *
from .data import *
from .inference import inference_warpper
# from .nano import *
# from .nanoprep import *

import argparse

parse_dict = {
    # 'preprocess': {
    #     'help': 'prepropcess the data',
    #     'func': preprocess_data,
    #     '--gtf': 'the path of GTF file',
    #     '--fasta': 'the path of the chromosome-level FASTA file that accompanies the GTF file',
    #     '--mode': ['full', 'mature', 'select the mode of preprocessing'],
    #     '--m6a_bed_dir': 'the directory containing the BED files that record the m6a sites',
    #     '--out': 'the path of preprocessed file',
    # }, 
    # 'train': {
    #     'help': 'train the model',
    #     'func': train_sramp,
    #     '--data': 'the path of preprocessed data',
    #     '--out': 'the path of trained model',
    # }, 
    'predict': {
        'help': 'predict the m6a sites from fasta',
        'func': inference_warpper,
        '--fasta': 'the path of fasta file',
        '--db': 'the dir of blast database',
        '--blast': 'the dir of blast bin folder',
        '--model': 'the path of trained model',
        '--out': 'the path of predicted results file',
    }, 
    # 'nanopreprocess': {
    #     'help': 'prepropcess the nanopore data',
    #     'func': preprocess_nano_data,
    #     '--file': 'the output file of nanopolish',
    #     '--out': 'the path of preprocessed file',
    # }, 
    # 'nanotrain': {
    #     'help': 'train the nanopore model',
    #     'func': train_nano_sramp,
    #     '--data': 'the path of preprocessed data',
    #     '--out': 'the path of trained model',
    # }, 
    # 'nanopred': {
    #     'help': 'predict the m6a sites using nanopore data',
    #     'func': predict_nano_sramp,
    # },
}

WELCOME = '''
Welcome to deepSRAMP!
----------------------
'''

def main():
    # print(WELCOME)
    parser = argparse.ArgumentParser(description='Welcome to deepSRAMP!')
    subparsers = parser.add_subparsers(required=True)
    
    for i in parse_dict:
        parser_a = subparsers.add_parser(i, help=parse_dict[i]['help'])
        for j in parse_dict[i]:
            if j in ['help', 'func']: continue
            if type(parse_dict[i][j]) == list:
                parser_a.add_argument(j, choices=parse_dict[i][j][:-1], help=parse_dict[i][j][-1], required=True)
            else:
                parser_a.add_argument(j, help=parse_dict[i][j], required=True)
        parser_a.set_defaults(func=parse_dict[i]['func'])
    
    args = parser.parse_args()
    args.func(args)

