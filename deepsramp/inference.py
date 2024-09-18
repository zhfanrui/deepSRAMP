import subprocess
import uuid
from io import StringIO

from . import *

def parse_afa(f, ref):
    seqs = read_seq(f)
    refseq, queseq = seqs[ref], seqs['query']
    
    ref2que = {}
    shift = 0
    for i in range(len(refseq)):
        if refseq[i] == '-':
            shift -= 1
        ref2que[i+shift] = i
    ref2que[1+i+shift] = i+1
        
    return refseq, queseq, ref2que
            
def blast(fa, name, db, blast_path):
    uid = f'{uuid.uuid1()}'
    sdf = utils.load(f'{db}.df')
    # echop = subprocess.Popen(['echo', '-e', fa], stdout=subprocess.PIPE)
    dbpath = f'{db}'
    # res = subprocess.check_output(['blast/bin/blastn', '-db', dbpath, '-max_target_seqs', '5', '-num_threads', '35', '-outfmt', '6 qacc sacc evalue length pident qstart qend sstart send btop'], stdin=echop.stdout)
    # if not res: return pd.DataFrame()
    res = b'''NM_000546.6	ENST00000620739	0.0	2512	100.000	1	2512	49	2560	2512
NM_000546.6	ENST00000269305	0.0	2512	100.000	1	2512	49	2560	2512
NM_000546.6	ENST00000619485	0.0	2509	99.880	4	2512	1	2506	110G-C-A-2396
NM_000546.6	ENST00000445888	0.0	2509	99.880	4	2512	1	2506	110G-C-A-2396
NM_000546.6	ENST00000610292	0.0	2296	100.000	217	2512	325	2620	2296
NM_000546.6	ENST00000610292	1.68e-104	207	100.000	10	216	1	207	207'''
    rdf = pd.read_csv(StringIO(res.decode()), comment='#', sep='\t', header=None)
    
    ref = rdf[1][0]
    refsdf = sdf.loc[ref].copy()
    refseq = refsdf.seq
    
    input_file = f'deepsramp_{uid}.fa'
    output_file = f'deepsramp_{uid}.afa'
    with open(input_file, 'w') as f:
        f.write(f'>{ref}\n{refseq}\n>query\n{fa}')
    subprocess.run([f'{blast_path}/muscle', '-align', input_file, '-output', output_file])
    refseq, queseq, d = parse_afa(output_file, ref)
    
    subprocess.run(['rm', '-f', input_file])
    subprocess.run(['rm', '-f', output_file])
    
    refsdf = sdf.loc[ref].copy()
    for i in ['splice', 'cds']:
        refsdf[i] = set(d.get(i) for i in refsdf.splice)
    refsdf['length'] = d.get(refsdf['length'])
    refsdf['refseq'] = refseq
    refsdf['seq'] = queseq
    refsdf.name = f'{name} to {ref}'
    refsdf['pos'] = get_drach(refsdf.seq, DRACH_PAT)

    evaldf = []
    for i in refsdf.pos:
        genome_pos = get_genome_pos(i, refsdf)
        gene_trans = sdf[sdf.id == refsdf.id]
        grp = refsdf[0] + refsdf[6] + str(genome_pos)
        label = (grp in utils.notnaor(refsdf.grp, [])) * 1
        evaldf += [(refsdf.name, i, grp, label)]
        
        for j in gene_trans.itertuples():
            trans_pos = get_mature_pos(genome_pos, j)
            if trans_pos != -1:
                evaldf += [(j.Index, trans_pos, grp, label)]
    
    evaldf = pd.DataFrame(evaldf, columns=['trans', 'pos', 'grp', 'label'])
    evaldf = evaldf.merge(pd.concat([sdf, pd.DataFrame(refsdf).T])[['cds', 'splice', 'length', 'min', 'max', 'seq', 'refseq']], left_on='trans', right_index=True, how='left')
    
    return evaldf

def inference(seqs, blast_path='blast/', db='hg38_mature', model_path='model/full_400_mature.model', lsep='\n'):
    seqs = seqs.split('>')
    evaldfs = []
    names = []
    for i in seqs:
        if i == '':
            continue
        else:
            seq = i.split(lsep)
            name = seq[0]
            names += [name]
            seq = ''.join(seq[1:])
            evaldf = blast(seq, name, db=db, blast_path=blast_path)
    
            evalds, grpidx = df2ds_multi(evaldf, return_grp=True)
            evaldl = ds2dl(evalds, drop_last=False, num_workers=2, shuffle=False)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MultiSRAMP().to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            preds, ys = pred_loop(evaldl, model, device=device)
            evaldf = evaldf[evaldf.trans.str.contains('to')].set_index('grp').loc[grpidx]
            evaldf['pred'] = preds

            evaldfs += [evaldf]
    evaldfs = pd.concat(evaldfs)
    return evaldfs

def inference_warpper(args):
    with open(args.fasta) as f:
        res = inference(f.read(),
                        db=args.db,
                        model_path=args.model,
                        blast_path=args.blast)
    res.to_csv(args.out)

