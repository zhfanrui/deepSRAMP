# deepSRAMP

## Installation

1. Install `conda`;
2. Clone this repo;
3. Install `deepSRAMP` through either

    A.
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







