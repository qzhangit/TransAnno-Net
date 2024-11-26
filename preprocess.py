#数据预处理
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse

panglao = sc.read_h5ad('/home/qzhang/Project/scBERT/data/PanglaoDB_Lung_hvg_16906_MAtrix_generename.h5ad')
data = sc.read_h5ad('/home/qzhang/Project/scBERT/raw10X_data/Control_finetune.h5ad')
counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)#创建一个稀疏矩阵来存储处理后的数据，大小为 (data.X.shape[0], panglao.X.shape[1])
ref = panglao.var_names.tolist()#获取变量（基因）名称列表
obj = data.var_names.tolist()

# gene_names = panglao.var_names

# # 随机选择 13000 个基因
# num_genes_to_select = 13000
# selected_genes = np.random.choice(gene_names, num_genes_to_select, replace=False)

# # 打印选中的基因
# print(selected_genes)

## 使用循环将 "your_raw_data" 中存在的基因的表达值复制到 "panglao_10000" 中相应的位置
for i in range(len(ref)):
    if ref[i] in obj:
        loc = obj.index(ref[i])
        counts[:,i] = data.X[:,loc]

## 转换为 CSR 格式的稀疏矩阵
counts = counts.tocsr()
# 创建一个新的 AnnData 对象，将 counts 作为数据矩阵
new = ad.AnnData(X=counts)
# 设置变量名称和观测名称
new.var_names = ref
new.obs_names = data.obs_names

# 复制观测信息和未注释信息（annotations）从 "your_raw_data" 到新的 AnnData 对象
new.obs = data.obs
new.uns = panglao.uns

# 对新的数据进行处理
sc.pp.filter_cells(new, min_genes=200)
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)

# 将处理后的数据写入新的 HDF5 文件
new.write('./data/preprocessed_Control_Lung_16906Matrix_afterpanglao.h5ad')


