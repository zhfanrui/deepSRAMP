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

1. Install `conda`;
2. Clone this repo;
3. Install `deepSRAMP` through either
    a. 
    A.
   ```sh
pip install deepsramp
   ```
   B.
```sh
conda create -y -n sramp ipykernel 
conda activate sramp
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torch pandas scikit-learn regex joblib tqdm matplotlib seaborn shap
```

    B. 
```sh
conda env create -f env.yml
```
5. Download GTF and FASTA files through `sh download.sh`;






```sh
rsync -av --exclude-from ../../deepsramp/exclude_file.txt ../../deepsramp/ .
```