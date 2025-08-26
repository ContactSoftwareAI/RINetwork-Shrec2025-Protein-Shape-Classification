# Shrec 2025 Protein Classification

Code from [SHREC2025 Protein Classification Challenge](https://shrec2025.drugdesign.fr/#envisioned-task) 
(announced for [3DOR2025](https://3dor.cs.ucl.ac.uk/home))

:tada: **News:** Preprint is [available](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5258950).

## Summary

Dataset size: 9,244 for training (with labels) + 2,321 for testing (unlabelled). Number of classes: 97.


We split training set into 7,428/1,816 (approx. 80:20) as training/validation split.


For every vtk file we extract point cloud with 8192 points.

Two [models](https://drive.contact.de/s/X9eiUArRXTTX1pT) are trained/tested:
+ RUN 1: Geometry-only model (uses only point cloud as an input),
+ RUN 2: Geometry+Potential Model (point cloud + additional features for every point: potential and normal potential).

All statistics on test set:

RUN 1:
Accuracy: 91.90%,
Precision: 80.65%,
Recall: 78.25%,
F1 Score: 78.59%.


RUN 2:
Accuracy: 92.76%,
Precision: 88.71%,
Recall: 84.72%,
F1 Score: 85.84%.


Details on the method can found [here.](https://github.com/ContactSoftwareAI/RINetwork-Shrec2025-Protein-Shape-Classification/blob/main/docu.pdf)


## Running the code

### How to reproduce results?
+ **Step 1:** Download preprocessed point cloud data from test set from [here](https://drive.contact.de/s/2uYAC96R0PnIHUR).

+ **Step 2:** Download pretrained models in ```log``` folder from [here](https://drive.contact.de/s/X9eiUArRXTTX1pT). 

+ **Step 3:** Prepare environment:
```
python3 -m venv .venv_gpu
source .venv_gpu/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_gpu.txt
```

+ **Step 4:** Run scripts with:
```
# activate environment
source .venv_gpu/bin/activate
# RUN 1:
 python test_classification_protein.py
# RUN 2:
 python test_classification_protein_features.py
```


### How to train a model?
Train models from scratch with:
```
# activate environment
source .venv_gpu/bin/activate
# RUN 1:
 python train_classification_protein.py
# RUN 2:
 python train_classification_protein_features2.py
```


### Download and prepare data
Download original (raw, unprocessed) [training set](https://shrec2025.drugdesign.fr/files/train_set.tar.xz) and [test set](https://shrec2025.drugdesign.fr/files/test_set.tar.xz).
Extract point clouds from .vtk files with script ```extract_pc_from_vtk\convert_extract_features.py```.


## Acknowledgments
Our code is build on top of the original [RIConv++ paper](https://arxiv.org/abs/2202.13094) and their [code repository.](https://github.com/cszyzhang/riconv2)

We thank authors for open-sourcing their code. If you use this code, please consider citing them.










