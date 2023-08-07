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

### Model Zoo
Supported Models:
- [Devign](https://github.com/epicosy/devign)
- [ReGVD](https://github.com/daiquocnguyen/GNN-ReGVD)
- [LineVul](https://github.com/awsm-research/LineVul/blob/main/linevul/linevul_main.py)
- VulDeePecker
- TextCNN
- VulCNN
- transformers

### Supported Datasets
- REVEAL
- FFMPEG + Qemu
- SARD/NVD
- MSR_20_CODE
- CODEXGLUE
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

### Quick Start

Getting started with ASTERIA Detection is simple and straightforward. Follow the installation instructions in the provided documentation, and you'll be ready to explore and customize the wide array of vulnerability detection models.

### Customization

ASTERIA Detection prides itself on its flexibility. Users can easily tailor the system to their specific needs, thanks to the comprehensive set of tools and configurations. Whether you are a seasoned security professional or new to vulnerability detection, ASTERIA provides an adaptable solution.

### Conclusion

ASTERIA Detection represents the pinnacle of vulnerability detection frameworks, offering unmatched extensibility, intelligence, and adaptability. Its ease of use, coupled with a rich selection of datasets, ensures that users can effortlessly navigate the complex terrain of secure computing.

For detailed guides, tutorials, and API references, please refer to the full documentation available in the repository.

Start exploring ASTERIA Detection today and take a step forward into a secure and resilient digital future.
