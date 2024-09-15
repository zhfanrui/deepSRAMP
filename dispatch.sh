# rm -f *.out
CUDA_VISIBLE_DEVICES=5 python $1 -f $2 -o $3 -t ythdf > ythdf.out &
CUDA_VISIBLE_DEVICES=6 python $1 -f $2 -o $3 -t sysy > sysy.out &
CUDA_VISIBLE_DEVICES=7 python $1 -f $2 -o $3 -t cd8t > cd8t.out &
CUDA_VISIBLE_DEVICES=8 python $1 -f $2 -o $3 -t a549 > a549.out &
CUDA_VISIBLE_DEVICES=9 python $1 -f $2 -o $3 -t molm > molm.out &
CUDA_VISIBLE_DEVICES=0 python $1 -f $2 -o $3 -t abcam > abcam.out &
CUDA_VISIBLE_DEVICES=1 python $1 -f $2 -o $3 -t hela > hela.out &
wait

# extra = ''
# sh dispatch.sh mds.py data/GSE/compare_mature_multi.data model
# sh dispatch.sh ds.py data/GSE/compare_mature.data model

# extra = 'random'
# sh dispatch.sh ds.py data/GSE/compare_mature_random.data model

