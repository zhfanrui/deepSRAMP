from setuptools import setup, find_packages

setup(
    name='deepsramp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'scikit-learn',
        'joblib',
        'tqdm',
        'matplotlib',
        'seaborn',
    ],
    author='Rui Fan',
    author_email='r.fan@bjmu.edu.cn',
    description='Mammalian m6A site predictor',
    license='MIT',
    url='https://github.com/zhfanrui/deepSRAMP'
)