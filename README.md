# TransAnno-Net
TransAnno-Net, a deep learning framework based on transfer learning and Transformer architecture, designed for efficient and accurate cell type annotations in large-scale scRNA-seq datasets of mouse lung organs.
# Install
Software version reference requirements.txt
# Diagram of TransAnno-Net framework

The workflow of this study is divided into four main steps: data preparation, data pre-processing, model pre-training and fine-tuning specific to the cell type annotation task, and performance evaluation. In the first step, data from different sources are merged into a unified file, and in the second step, the model receives the gene expression matrix of the scRNA-seq data as input and preprocesses the data using SCANPY, which includes filtering of low-quality cells, normalization, as well as log transformations and selection of highly-variable genes. In the third step, TransAnno-Net is pre-trained on an unlabeled large-scale scRNA-seq dataset to identify potential features of cell type representations, which helps to eliminate batch effects between datasets. It is then fine-tuned on several specific, manually labeled target datasets. Finally, the model performance is thoroughly evaluated to verify the usability and generalizability ability of the model.
![Workflow of automated cell type annotation of mouse lung tissue using TransAnno-Net. (A) Data preparation. (B) Preprocessing of gene expression matrix data for quality control, normalization, and selection of highly variable genes. (C) The preprocessed data are fed into TransAnno-Net for embedding representation and classification of cell types. The training process of TransAnno-Net consists of two stages: self-supervised learning on unlabeled data to obtain a pre-trained model, and supervised learning on a cell type-specific labeling task to obtain a fine-tuned model. (D) Performance evaluation of TransAnno-Net on fine-tuned and unseen datasets.](https://github.com/qzhangit/TransAnno-Net/blob/main/Picture/framework.png)

# Data
The data can be downloaded from these links.
https://panglaodb.se/
https://www.ncbi.nlm.nih.gov/geo/
https://cblast.gao-lab.org/
