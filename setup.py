from setuptools import setup, find_packages

setup(
    name='ASTERIA-Detection',
    version='1.0.1',
    description='A framework consists of SOTA Vulnerability Detection Models and Datasets.',
    url='https://github.com/Asteriska001/ASTERIA-Detection',
    author='Asteriska001',
    author_email='asteriska001@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
    'pyyaml',
    'tabulate',
    #'torch_geometric==2.1.0',
    #'scipy==1.7.0',
    'omegaconf',
    'pandas',
    'networkx',
    'gensim',
    #'torch-sparse==0.6.12',
    #'torch-scatter==2.1.1',
    #'cuml',
    'antlr4-tools',
    'antlr4-python3-runtime==4.13.1',
    'tree_sitter',
    'programl',
   # 'dgl==0.6.1'
    ]
)
