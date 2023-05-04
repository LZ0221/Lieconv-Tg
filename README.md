# Lieconv-Tg
This is the code repo for the paper Large-scale Glass Transition Temperature Prediction with Equivariant Graph Neural Network for Screening Polymers .We collected most of the homopolymer data from Polyinfo and further processed it to obtain a larger dataset containing 7246 homopolymer data.We used this dataset to build a neural network based on Lieconv to predict the glass transition temperature of the polymer.Using the same dataset, we also constructed an Image-CNN model and an Eccconv-based model. In contrast, the Lieconv-based model has better predictive and derivative power and is also used for subsequent screening of promising candidates from a benchmark database, named PI1M.

Package required:
==
We recommend to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [pip](https://pypi.org/project/pip/).

Lieconv
--
* [Python3](https://www.python.org/)
* [RDKit](https://rdkit.org/)
* [PyTorch](https://pytorch.org/get-started/locally/)

Image-CNN
--
* [Python3](https://www.python.org/)
* [RDKit](https://rdkit.org/)
* [TensorFlow](https://www.tensorflow.org/?hl=zh-cn)
* [DeepChem](https://deepchem.readthedocs.io/en/latest/)

Eccconv
--
* [Python3](https://www.python.org/)
* [RDKit](https://rdkit.org/)
* [TensorFlow](https://www.tensorflow.org/?hl=zh-cn)
* [Spektral](https://graphneural.network/)

By using the [requirements/conda/lieconv-environment.yml](https://github.com/LZ0221/Lieconv-Tg/blob/main/requirements/conda/lieconv-environment.yml), requirements/pip/requirements.txt file, it will install all the required packages(Take Lieconv as an example).
```
git clone https://github.com/zmzhang/SigmaCCS.git
cd Lieconv-Tg
conda env create -f requirements/conda/lieconv-environment.yml
conda activate lieconv
```
