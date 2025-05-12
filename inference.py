import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py
from utils_data.dataset import Path_ST_Dataset, calculate_st_embedding
import scanpy as sc
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import glob
from config import get_config
from model.models import CLIPModel_ViT_itm_v14_MSE

def save_embedding_all_sample(cancer, model_path, model, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = os.path.join(args.data_path, str(cancer), 'adata_HVGs')
    for sample in os.listdir(path):
        sample = sample.split('.')[0]
        dataset = Path_ST_Dataset(
            data_path=args.data_path,
            cancer=cancer,
            sample_id=sample,
            mode='test'
            )
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        image_embedding, spot_embedding = get_embeddings(model_path=model_path, model=model, loader=data_loader)
        image_embedding, spot_embedding = image_embedding.cpu().numpy(), spot_embedding.cpu().numpy()
        np.save(os.path.join(save_path, 'image_embedding_' + str(sample) + '.npy'), image_embedding)
        np.save(os.path.join(save_path, 'spot_embedding_' + str(sample) + '.npy'), spot_embedding)

def get_embeddings(model_path, model, loader):

    state_dict = torch.load(model_path)['model_state_dict']
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    image_embeddings_list = []
    spot_embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_features = image_features.mean(dim=1)
            image_embeddings = model.image_projection(image_features)
            image_embeddings_list.append(image_embeddings)

            spot_embeddings_list.append(model.spot_projection(batch["st_data"].cuda()))

    return torch.cat(image_embeddings_list), torch.cat(spot_embeddings_list)

def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()

if __name__ == '__main__':

    args = get_config()
    embedding_length = calculate_st_embedding(args=args)
    adata_file_name = 'adata_HVGs'

    result_dict = []
    All_dataset_mean_HEG = []
    All_dataset_mean_HVG = []
    All_dataset_mean_marker_genes = []

    for cancer in args.cancer_list:
        HEG_k = []
        HVG_k = []
        split_file_list = glob.glob(os.path.join(args.data_path, str(cancer), 'splits', '*.csv'))
        k = int(len(split_file_list) / 2)
        for k_i in range(k):
            model_path = os.path.join(args.save_dir, str(cancer), 'pth_'+str(k_i), 'best_model.pth')
            save_path = os.path.join( './result', str(cancer), 'pth_'+str(k_i), 'embeddings/')
            model = CLIPModel_ViT_itm_v14_MSE(spot_embedding=embedding_length[str(cancer)]).cuda()
            save_embedding_all_sample(cancer=cancer, model_path=model_path, model=model, save_path=save_path)

            test_df = pd.read_csv(os.path.join(args.data_path, str(cancer), 'splits', 'test_'+str(k_i)+'.csv'))
            test_list = list(test_df.iloc[:]['sample_id'])

            train_df = pd.read_csv(os.path.join(args.data_path, str(cancer), 'splits', 'train_'+str(k_i)+'.csv'))
            train_list = list(train_df.iloc[:]['sample_id'])

            spot_key = []
            expression_key = []
            for sample in train_list:
                spot_embedding = np.load(os.path.join(save_path, 'spot_embedding_' + str(sample) + '.npy'))
                if len(spot_key) == 0:
                    spot_key = spot_embedding
                else:
                    spot_key = np.concatenate((spot_key, spot_embedding), axis=0)

                spot_expression = sc.read_h5ad(os.path.join(args.data_path, str(cancer), adata_file_name, str(sample) + '.h5ad'))
                patch_path = os.path.join(args.data_path, str(cancer), 'patches', str(sample) + '.h5')
                with h5py.File(patch_path, 'r') as f:
                    barcodes_all = f['barcode'][:].flatten().astype(str).tolist()
                st_expression = spot_expression[barcodes_all][:].to_df().to_numpy()
                if len(expression_key) == 0:
                    expression_key = st_expression
                else:
                    expression_key = np.concatenate((expression_key, st_expression), axis=0)

            result_mean_HEG = []
            result_mean_HVG = []
            method = 'average'

            for sample in test_list:
                image_query = np.load(os.path.join(save_path, 'image_embedding_' + str(sample) + '.npy'))

                spot_expression = sc.read_h5ad(os.path.join(args.data_path, str(cancer), adata_file_name, str(sample) + '.h5ad'))
                patch_path = os.path.join(args.data_path, str(cancer), 'patches', str(sample) + '.h5')
                with h5py.File(patch_path, 'r') as f:
                    barcodes_all = f['barcode'][:].flatten().astype(str).tolist()
                    coords = f['coords'][:]
                expression_df = spot_expression[barcodes_all][:].to_df()
                genes_list = list(expression_df.columns.values)
                expression_gt = spot_expression[barcodes_all][:].to_df().to_numpy()

                # -------------------
                if method == "average":
                    print("finding matches, using average of top 50 expressions")
                    indices = find_matches(spot_key, image_query, top_k=50)
                    matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
                    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
                    for i in range(indices.shape[0]):
                        matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0)
                        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

                    print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
                    print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

                true = expression_gt
                pred = matched_spot_expression_pred

                # genewise correlation
                corr = np.zeros(pred.shape[0])
                for i in range(pred.shape[0]):
                    corr[i] = np.corrcoef(pred[i, :], true[i, :], )[0, 1]
                corr[np.isnan(corr)] = 0.0
                print("Mean correlation across cells: ", np.mean(corr))

                corr = np.zeros(pred.shape[1])
                for i in range(pred.shape[1]):
                    corr[i] = np.corrcoef(pred[:, i], true[:, i], )[0, 1]
                corr[np.isnan(corr)] = 0.0
                print("max correlation: ", np.max(corr))
                ind_HEG = np.argsort(np.sum(true, axis=0))[-50:]
                print("mean correlation highly expressed genes: ", np.mean(corr[ind_HEG]))
                result_mean_HEG.append(np.mean(corr[ind_HEG]))
                ind_HVG = np.argsort(np.var(true, axis=0))[-50:]
                print("mean correlation highly variable genes: ", np.mean(corr[ind_HVG]))
                result_mean_HVG.append(np.mean(corr[ind_HVG]))

            HEG_k.append(float(np.mean(result_mean_HEG)))
            HVG_k.append(float(np.mean(result_mean_HVG)))

        mean_cancer_HEG = np.mean(HEG_k)
        mean_cancer_HVG = np.mean(HVG_k)

        print('{}__HEG__{}k_result:{}'.format(str(cancer), k, mean_cancer_HEG))
        print('{}__HVG__{}k_result:{}'.format(str(cancer), k, mean_cancer_HVG))
