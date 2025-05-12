import os
import scanpy as sc
import argparse


def get_config():
    parser = argparse.ArgumentParser(description='config for data pre_process')
    parser.add_argument('--data_path', type=str, default=r'./Data/public/HEST', help='')
    parser.add_argument('--cancer_list', type=list, default=['READ', 'COAD'], help='')
    config = parser.parse_args()
    return config

def HVG_selection(n_top_genes, cancer_name):
    original_st_paths = os.path.join(args.data_path, cancer_name, 'adata')
    common_gene_name = []
    for sample in os.listdir(original_st_paths):
        adata = sc.read_h5ad(os.path.join(original_st_paths, sample))
        gene_name_list = adata.to_df().columns.to_list()
        if len(common_gene_name) == 0:
            common_gene_name = gene_name_list
        else:
            common_gene_name = list(set(common_gene_name) & set(gene_name_list))
    gene_name_union = []
    for sample in os.listdir(original_st_paths):
        print(sample)
        adata = sc.read_h5ad(os.path.join(original_st_paths, sample))
        adata = adata[:, common_gene_name]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        highly_variable_genes = adata.var.index[adata.var['highly_variable']].tolist()
        gene_name_union = list(set(gene_name_union) | set(highly_variable_genes))

    print("Number of HVGs: ", len(gene_name_union))
    for sample in os.listdir(original_st_paths):
        adata = sc.read_h5ad(os.path.join(original_st_paths, sample))
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata = adata[:, gene_name_union]  # 将高变基因筛选出来
        save_dir = os.path.join(args.data_path, cancer_name, 'adata_HVGs')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sc.write(filename=os.path.join(save_dir, sample), adata=adata)

if __name__ == '__main__':

    args = get_config()
    for cancer in args.cancer_list:
        print("preprocess：", cancer)
        HVG_selection(n_top_genes=1000, cancer_name=cancer)






