# deepSRAMP

SRAMP (http://www.cuilab.cn/sramp) is a popular mammalian m6A site predictor we previously developed (Nucleic Acids Res 2016). SRAMP has been totally cited by more than 570 papers (google scholar, 4-16, 2024) and represents the mostly used algorithm in this field. A large number of m6A sites were identified by the helps of SRAMP. After ~8 years after its development, Now we released deepSRAMP, which is designed based on a combined framework of Transformer neural network and recurrent neural network by fusing the sequence and genomic position features. The results showed that SRAMP2 greatly outperforms its predecessor SRAMP with 14.2 increase of AUC and 26.8 increase of AUPRC, and greatly outperforms other state-of-the-art m6A predictors (WHISTLE and DeepPromise) with 15.0% and 17.2 increase of AUC and 38.7% and 41.2% increase of AUPRC, respectively.

## Requirements

- torch
- pandas
- scikit-learn
- joblib
- tqdm
- matplotlib
- seaborn
- shap

## Installation

1. Install `conda` and create a virtual enviroument named `sramp` with `python` installed;
```sh
conda create -y -n sramp python 
conda activate sramp
```
2. Install `deepSRAMP` through
```sh
pip install deepsramp
```
3. Clone this repo, especially for `data` and `model` folder;
4. Download GTF and FASTA files through `sh download.sh`;

## Usage


## Citation

paper

## Tutorials

The reproduction of figures in the paper can be found in `ipynb` files.
- [Figure 1](fig1.ipynb)
- [Figure 2](fig2.ipynb)
- [Figure 3](fig3.ipynb)
- [Figure 4](fig4.ipynb)

## Performance

### Cross Validation
![](fig/fig1b_roc_mature.svg)
![](fig/fig1b_prc_mature.svg)

### Test on YTHDF1/2
![](fig/fig2_ythdf_roc.svg)
![](fig/fig2_ythdf_prc.svg)

### Test on m6Aatlas
![](fig/atlas2_human_prc.svg)



```sh
rsync -av --exclude-from ../../deepsramp/exclude_file.txt ../../deepsramp/ .
```