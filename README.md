# TransAnno-Net
A Deep Learning Framework Based on Migration Learning and Transformer Architecture
# Install
Software version reference requirements.txt <br>
torch==1.8.1+cu111 <br>
torchvision==0.9.1+cu111 <br>
transformers==4.31.0 <br>
scanpy==1.9.1 <br>
scikit-learn==1.3.2 <br>
scipy==1.9.0 <br>
numpy==1.22.4 <br>
pandas==1.5.1
# Diagram of TransAnno-Net framework

TransAnnoNet is a deep learning framework based on migration learning and Transformer architecture designed to provide efficient and accurate cell type annotation for large-scale scRNA-seq datasets of mouse lung organs.There are four main steps: data preparation, data preprocessing, pre-training and fine-tuning of the model for the cell type annotation task, and performance evaluation. <br>
<br>
The diagram and workflows of TranAnno-Net framework is shown as below.
![The diagram and workflows of TranAnno-Net framework is shown as below](https://github.com/qzhangit/TransAnno-Net/blob/main/Picture/framework.png)

# Data
The data can be downloaded from these links.
https://panglaodb.se/
https://www.ncbi.nlm.nih.gov/geo/
https://cblast.gao-lab.org/

| Dataset | Species | Organ |Number of cell types | cell numbers | Gene numbers |Protocols | Accession ID |
| :---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| PanglaoDB | Mouse | Lung | \ | 100024 | 45549 | microwell-seq, 10x chromium, Smart-seq2, drop-seq, SMART-seq2 | (https://panglaodb.se/) |
| GSE267861 | 〃 | 〃 | 12 | 24520 | 32285 | 10x chromium | GSE267861 |
| GSE264032 | 〃 | 〃 | 16 | 31928 | 32285 | 10x chromium | GSE264032 |
| ALIGNED_Mm_Lung (AML) | 〃 | 〃 | 17 | 16599 | 25174 | unknown | (https://cblast.gao-lab.org/) |
| Quake_10x_Lung (Q10xL) | 〃 | 〃 | 13 | 5449 | 23341 | 10x chromium | 〃 |
| ALIGNED_Mm_Trachea (AMT) | 〃 | Trachea | 18 | 12619 | 33948 | unknown |〃 |
| ALIGNED_Mm_Kidney (AMK) | 〃 | Kidney | 37 | 63659 | 35210 | unknown | 〃 |
| ALIGNED_Mm_Pancreas (AMP) | 〃 | Pancreas | 22 | 3450 | 25410 | unknown | 〃 |
| Plasschaert (PT) | 〃 | Trachea | 8 | 6977 | 28205 | inDrop | 〃 |
| Baron_mouse (BMP) | 〃 | Pancreas | 13 | 1886 | 14877 | inDrop | 〃 |

