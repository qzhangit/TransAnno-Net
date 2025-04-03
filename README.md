# TransAnno-Net
A Deep Learning Framework Based on Migration Learning and Transformer Architecture
# Install
Software version reference requirements.txt <br>
```python
pip --no-cache-dir install -r requirements.txt
```
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
![The diagram and workflows of TranAnno-Net framework is shown as below](https://github.com/qzhangit/TransAnno-Net/blob/main/Picture/framework.png) <br>
In the first step of this,we merge data from different sources into a unified file; in the second step, the model receives the gene expression matrix of the scRNA-seq data as input and preprocesses the data, including filtering of low-quality cells, normalization as well as logarithmic transformation and selection of highly variable genes. In the third step, TransAnno-Net is pre-trained on unlabeled large-scale scRNA-seq datasets to identify potential features of cell type representation, which helps to eliminate batch effects between datasets. It is then fine-tuned on several specific artificially labeled target datasets. Finally, the model performance was thoroughly evaluated to validate the usability and generalizability of the model.

# How to use the TransAnno-Net?
0.fully tested with Ubuntu 18.04 LTS, Python 3.8 with PyTorch 1.8.1 as the backend in a server equipped with Nvidia GTX 3090 GPUs <br>

1.clone the repo to local directory
```python
git clone https://github.com/qzhangit/TransAnno-Net.git
```
2.prepare the training and test dataset
* Prepare your gene `expression matrix`, convert it to `h5ad` format and process it with the `preprocess.py` file. <br>
* `PanglaoDB_Lung.h5ad` file is the selected gene, which can be downloaded through this link [https://pan.baidu.com/](https://pan.baidu.com/s/1yub8oxqsqzdLu_CK7ovUpg) with the access code “vyga”. <br>

3.start fintune training
```python
nohup python -m torch.distributed.launch --nproc_per_node=n Raw_finetune.py > nohup.out &  # n is the number of GPUs.
```
4.predictions of TransAnno-Net
```python
python predict.py
```
And congratulations, you have just used TransAnno-Net for your own data! Please feel free to let us know if you have any questions.

# Data
The data can be downloaded from these links. <br>
https://panglaodb.se/ <br>
https://www.ncbi.nlm.nih.gov/geo/ <br>
https://cblast.gao-lab.org/
<br>
| Dataset | Species | Organ |Number of cell types | cell numbers | Gene numbers |Protocols | Accession ID |
| :---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| PanglaoDB | Mouse | Lung | \ | 100,024 | 45,549 | microwell-seq, 10x chromium, Smart-seq2, drop-seq, SMART-seq2 | (https://panglaodb.se/) |
| GSE267861 | 〃 | 〃 | 12 | 24,520 | 32,285 | 10x chromium | [GSE267861](https://www.ncbi.nlm.nih.gov/geo/) |
| GSE264032 | 〃 | 〃 | 16 | 31,928 | 32,285 | 10x chromium | [GSE264032](https://www.ncbi.nlm.nih.gov/geo/) |
| ALIGNED_Mm_Lung (AML) | 〃 | 〃 | 17 | 16,599 | 25,174 | unknown | (https://cblast.gao-lab.org/) |
| Quake_10x_Lung (Q10xL) | 〃 | 〃 | 13 | 5449 | 23,341 | 10x chromium | 〃 |
| ALIGNED_Mm_Trachea (AMT) | 〃 | Trachea | 18 | 12,619 | 33,948 | unknown |〃 |
| ALIGNED_Mm_Kidney (AMK) | 〃 | Kidney | 37 | 63,659 | 35,210 | unknown | 〃 |
| ALIGNED_Mm_Pancreas (AMP) | 〃 | Pancreas | 22 | 3450 | 25,410 | unknown | 〃 |
| Plasschaert (PT) | 〃 | Trachea | 8 | 6977 | 28,205 | inDrop | 〃 |
| Baron_mouse (BMP) | 〃 | Pancreas | 13 | 1886 | 14,877 | inDrop | 〃 |
| Lung(Human) | Human | Lung | 9 | 39,778 | \ | 10x chromium | (https://doi.org/10.6084/m9.figshare.11981034.v1) |

# Time Cost
On a server running Ubuntu 20.04 with an Intel(R) Xeon(R) Gold 6271C processor, two NVIDIA GeForce GTX 4090 graphics cards, and 256GB of RAM, processing 17,000 cells takes 24 minutes.
# Who are we?
TransAnno-Net is proposed and maintained by researchers from [WIT](https://www.wit.edu.cn/).

