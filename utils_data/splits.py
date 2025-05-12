import json
import os
import numpy as np
import pandas as pd
import random
import h5py
import scanpy as sc
import argparse
from sklearn.model_selection import KFold

def get_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'./Data/public/HEST',
                        help='')
    parser.add_argument('--cancer_list', type=list, default=['COAD', 'READ'], help='')
    config = parser.parse_args()
    return config

def split_train_val(args, save_path, cancer_list):
    for cancer in cancer_list:
        for sample in os.listdir(os.path.join(args.data_path, cancer, 'patches')):
            if str(sample).split('.')[-1] == 'h5':
                with h5py.File(os.path.join(args.data_path, cancer, 'patches', str(sample))) as f:
                    barcode_list = f['barcode'][:].flatten().astype(str).tolist()
                    random.shuffle(barcode_list)
                    train_barcode_list = barcode_list[: int(len(barcode_list) * 0.8)]
                    val_barcode_list = barcode_list[int(len(barcode_list) * 0.8):]
                    dict_train_val = {}
                    dict_train_val['train'] = train_barcode_list
                    dict_train_val['val'] = val_barcode_list
                    with open(os.path.join(save_path, cancer, 'splits', str(sample).split('.')[0] + '_train_val.json'), 'w', encoding='utf-8') as fp:
                        json.dump(dict_train_val, fp, ensure_ascii=False)
                    print(len(train_barcode_list), len(val_barcode_list))

def make_split_k(args):
    for cancer in args.cancer_list:
        sample_list = []
        for sample in os.listdir(os.path.join(args.data_path, str(cancer), 'adata')):
            sample_list.append(sample.split('.')[0])

        random.shuffle(sample_list)  # 确保划分随机性

        k = min(len(sample_list), 5)
        kf = KFold(n_splits=k)

        for i, (train_index, test_index) in enumerate(kf.split(sample_list)):
            train_samples = [sample_list[idx] for idx in train_index]
            test_samples = [sample_list[idx] for idx in test_index]

            train_dict = {
                'sample_id': train_samples,
                'patches_path': ['patches/' + str(s) + '.h5' for s in train_samples],
                'expr_path': ['adata/' + str(s) + '.h5ad' for s in train_samples]
            }
            df_train = pd.DataFrame(train_dict)

            test_dict = {
                'sample_id': test_samples,
                'patches_path': ['patches/' + str(s) + '.h5' for s in test_samples],
                'expr_path': ['adata/' + str(s) + '.h5ad' for s in test_samples]
            }
            df_test = pd.DataFrame(test_dict)

            save_path = os.path.join(args.data_path, str(cancer), 'splits')
            os.makedirs(save_path, exist_ok=True)
            df_train.to_csv(os.path.join(save_path, f'train_{i}.csv'), sep=',', index=False)
            df_test.to_csv(os.path.join(save_path, f'test_{i}.csv'), sep=',', index=False)


if __name__ == '__main__':
    args = get_config()
    make_split_k(args=args)
    split_train_val(args=args, save_path=args.data_path, cancer_list=args.cancer_list)




