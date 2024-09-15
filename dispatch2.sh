# rm -f *.out
CUDA_VISIBLE_DEVICES=5 python $1 -f $2 -o $3 -t $4 -m full > full.out &
CUDA_VISIBLE_DEVICES=8 python $1 -f $2 -o $3 -t $4 -m genomeonly > genomeonly.out &
CUDA_VISIBLE_DEVICES=9 python $1 -f $2 -o $3 -t $4 -m seqonly > seqonly.out &
wait

# sh dispatch2.sh mds.py data/sramp1/sramp1_mature.data model mature
# sh dispatch2.sh mds.py data/sramp1/sramp1_full.data model full

