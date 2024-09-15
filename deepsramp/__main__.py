from scripts import *
import argparse

parse_dict = {
    'preprocess': {
        'help': 'prepropcess the data',
        'func': preprocess_data,
        '--gtf': 'the path of GTF file',
        '--fasta': 'the path of the chromosome-level FASTA file that accompanies the GTF file',
        '--mode': ['full', 'mature', 'select the mode of preprocessing'],
        '--m6a_bed_dir': 'the directory containing the BED files that record the m6a sites',
        '--out': 'the path of preprocessed file',
    }, 
    'train': {
        'help': 'train the model',
        'func': train_sramp,
        '--data': 'the path of preprocessed data',
        '--out': 'the path of trained model',
    }, 
    'pred': {
        'help': 'predict the m6a sites',
        'func': predict_sramp,
        '--model': 'the path of trained model',
        '--out': 'the path of predicted results',
    }, 
    'nanopreprocess': {
        'help': 'prepropcess the nanopore data',
        'func': preprocess_nano_data,
        '--file': 'the output file of nanopolish',
        '--out': 'the path of preprocessed file',
    }, 
    'nanotrain': {
        'help': 'train the nanopore model',
        'func': train_nano_sramp,
        '--data': 'the path of preprocessed data',
        '--out': 'the path of trained model',
    }, 
    'nanopred': {
        'help': 'predict the m6a sites using nanopore data',
        'func': predict_nano_sramp,
    },
}

WELCOME = '''
 .oooooo..o ooooooooo.         .o.       ooo        ooooo ooooooooo.        .oooo.         .oooo.   
d8P'    `Y8 `888   `Y88.      .888.      `88.       .888' `888   `Y88.    .dP""Y88b       d8P'`Y8b  
Y88bo.       888   .d88'     .8"888.      888b     d'888   888   .d88'          ]8P'     888    888 
 `"Y8888o.   888ooo88P'     .8' `888.     8 Y88. .P  888   888ooo88P'         .d8P'      888    888 
     `"Y88b  888`88b.      .88ooo8888.    8  `888'   888   888              .dP'         888    888 
oo     .d8P  888  `88b.   .8'     `888.   8    Y     888   888            .oP     .o .o. `88b  d88' 
8""88888P'  o888o  o888o o88o     o8888o o8o        o888o o888o           8888888888 Y8P  `Y8bd8P'  

----------------------------------------------------------------------------------------------------
'''

def main():
    print(WELCOME)
    parser = argparse.ArgumentParser(description='Welcome to SRAMP2!')
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
    print(args)
    args.func(args)
    
main()


