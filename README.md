# ASTERIA Detection: Architecture for Security Testing with Extensible Resourceful Intelligence and Adaptation

## Deeplearning-based Vulnerability Detection Framework

### Introduction

ASTERIA Detection aims to transform the landscape of security in computing systems by providing a comprehensive framework for vulnerability detection. This cutting-edge architecture relies on deep learning methodologies, offering extensible and customizable solutions at the forefront of technology.

### Key Features

- **Extensible Resourceful Intelligence**: Incorporates AI and machine learning algorithms that can be easily extended and adapted to various security testing scenarios.
- **Rich and Diverse Data Preprocessing Mechanisms**: Including code representation methods, graph neural networks, and sequence representation methods, allowing the free combination of preprocessing steps and adjustment of the processing flow.
- **Adaptable Framework**: Tailored to meet individual requirements with built-in support for various vulnerability detection models.
- **Abundant Datasets**: Offers a wide variety of datasets to train and test the models, providing a versatile environment for experimentation.
- **Ease of Use**: Designed with the user in mind, ASTERIA offers an intuitive interface that makes implementing and modifying state-of-the-art (SOTA) vulnerability detection models a breeze.

### Quick Start
Getting started with ASTERIA Detection is simple and straightforward. Follow the installation instructions in the provided documentation, and you'll be ready to explore and customize the wide array of vulnerability detection models.
#### Required environment
- Python3 && PIP installed
- Linux/Windows/macOS
- Torch + CUDA enabled


#### Conda Installation
Type the following commands in your Shell
```
conda create -n ASTERIA && conda activate ASTERIA
git clone https://github.com/Asteriska001/ASTERIA-Detection
cd ASTERIA-Detection && chmod +x ./install.sh
./install.sh
```

#### Direct Installation
Type the following commands in your Shell
```
git clone https://github.com/Asteriska001/ASTERIA-Detection
cd ASTERIA-Detection && chmod +x ./install.sh
./install.sh
```
#### Docker Implementation
` TODO: docker run`

### Usage Example
This framework is primarily used through configuration files written in YAML syntax. 

These configuration files contain information about the model, dataset, metrics , preprocessing, as well as configurations required for training, validation, and inference. 

You can conveniently utilize it through **Jupyter Notebook** or opt for a **command-line** method.

We've already included several notebook files in the 'notebooks' directory for your reference. 
You're welcome to delve deeper into them.
#### Jupyter Notebook
Navigate to the 'notebooks' directory and use Jupyter in the way you're accustomed to.

#### Command-Line:
We take the ReGVD method paired with the REVEAL dataset as an example.
```
# Training
python tools/train.py --cfg configs/regvd_reveal.yaml

# Validation
python tools/val.py --cfg configs/regvd_reveal.yaml
```
And we have already provided some usable examples in the 'config' directory.
You can further inspect/check it.
### Model Zoo
Supported Models/Modules:
- [Devign](https://github.com/epicosy/devign)
- [ReGVD](https://github.com/daiquocnguyen/GNN-ReGVD)
- [LineVul](https://github.com/awsm-research/LineVul/blob/main/linevul/linevul_main.py)
- VulDeePecker
- IVDetect
- LineVD
- ReGVD
- DeepWuKong
- TextCNN
- VulCNN
- transformers
- VulBERTa

### Supported Datasets
- REVEAL
- FFMPEG + Qemu
- SARD/NVD
- MSR_20_CODE/BigVul
- CODEXGLUE
- D2A
- ...

### Supported Preprocess Methods
- "Normalize"
- "PadSequence"
- "OneHotEncode"
- 'Tokenize'
- 'VocabularyMapping'
- 'LengthNormalization'
- 'Shuffle'
- ...
  
### Supported Representations Methods
- **AST Graph extractor / AST Graph builder**
- **LLVM Graph extractor / builder**
- **Syntax extractor / builder**
- **Vectorizers/word2vec**
- **sent2vec**
- ...


### Customization

ASTERIA Detection prides itself on its flexibility. Users can easily tailor the system to their specific needs, thanks to the comprehensive set of tools and configurations. Whether you are a seasoned security professional or new to vulnerability detection, ASTERIA provides an adaptable solution.

### Conclusion

ASTERIA Detection represents the pinnacle of vulnerability detection frameworks, offering unmatched extensibility, intelligence, and adaptability. Its ease of use, coupled with a rich selection of datasets, ensures that users can effortlessly navigate the complex terrain of secure computing.

For detailed guides, tutorials, and API references, please refer to the full documentation available in the repository.

Start exploring ASTERIA Detection today and take a step forward into a secure and resilient digital future.
