# Shrec2025ProteinClassification

Code from [SHREC2025 Protein Classification Challenge](https://shrec2025.drugdesign.fr/#envisioned-task)
(announced for [3DOR2025](https://3dor.cs.ucl.ac.uk/home))


Dataset size: 9,244 for training (with labels) + 2,321 for testing (unlabelled). Number of classes: 97.


We split training set into 7,428/1,816 (approx. 80:20) as training/validation split.


For every vtk file we extract point cloud with 8192 points.

Two models are trained/tested:
+ RUN 1: Geometry-only model (uses only point cloud as an input),
+ RUN 2: Geometry+Potential Model (point cloud + additional features for every point: potential and normal potential).


<!--Validation accuracy:-->
<!--+ RUN 1: 92.93%-->
<!--+ RUN 2: 93.97%-->



<!--Test accuracy:-->
<!--+ RUN 1: 91.90%-->
<!--+ RUN 2: 92.76%-->


All statistics on test set:

RUN 1:
Accuracy: 0.9190,
Precision: 0.8065,
Recall: 0.7825,
F1 Score: 0.7859


RUN 2:
Accuracy: 0.9276,
Precision: 0.8871,
Recall: 0.8472,
F1 Score: 0.8584


## Acknowledgments:
Our code is build on top of the original RIConv++ paper and their code repository: https://github.com/cszyzhang/riconv2

We thank authors for open-sourcing their code.
