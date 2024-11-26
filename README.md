# TransAnno-Net
TransAnno-Net, a deep learning framework based on transfer learning and Transformer architecture, designed for efficient and accurate cell type annotations in large-scale scRNA-seq datasets of mouse lung organs.
# Install
Software version reference requirements.txt
# Diagram of TransAnno-Net framework

The workflow of this study is divided into four main steps: data preparation, data pre-processing, model pre-training and fine-tuning specific to the cell type annotation task, and performance evaluation. In the first step, data from different sources are merged into a unified file, and in the second step, the model receives the gene expression matrix of the scRNA-seq data as input and preprocesses the data using SCANPY, which includes filtering of low-quality cells, normalization, as well as log transformations and selection of highly-variable genes. In the third step, TransAnno-Net is pre-trained on an unlabeled large-scale scRNA-seq dataset to identify potential features of cell type representations, which helps to eliminate batch effects between datasets. It is then fine-tuned on several specific, manually labeled target datasets. Finally, the model performance is thoroughly evaluated to verify the usability and generalizability ability of the model.
![Image text](https://github.com/qzhangit/TransAnno-Net/blob/main/Picture/framework.png)

# Data
The data can be downloaded from these links.
https://panglaodb.se/
https://www.ncbi.nlm.nih.gov/geo/
https://cblast.gao-lab.org/

| Dataset | Species | Organ |Number of cell types | cell numbers | Gene numbers |Protocols | Accession ID |
| :---- | :----: | ----: | :---- | :----: | ----: | :---- | :----: |
| PanglaoDB | 单元格2 | 单元格3 | 单元格1 | 单元格2 | 单元格3 | 单元格1 | 单元格2 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
| 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 | 单元格6 | 单元格4 | 单元格5 |
