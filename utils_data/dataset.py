import h5py
import scanpy as sc
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import random
import torch
import json
from torch.utils.data import DataLoader

class Path_ST_Dataset(Dataset):
    def __init__(self, data_path, cancer, sample_id, mode='train'):
        self.data_path = data_path
        self.cancer = cancer
        self.mode = mode
        self.patch_h5_path = os.path.join(self.data_path, self.cancer, 'patches', sample_id + '.h5')
        self.st_data_path = os.path.join(self.data_path, self.cancer, 'adata_HVGs', sample_id + '.h5ad')

        self.image_dict = {}
        with h5py.File(self.patch_h5_path, 'r') as f:
            self.barcodes_all = f['barcode'][:].flatten().astype(str).tolist()
            pathology_image = f['img'][:]
        for i, barcode in enumerate(self.barcodes_all):
            self.image_dict[barcode] = pathology_image[i, :, :, :]
        self.adate = sc.read_h5ad(self.st_data_path)
        if self.mode == 'test':
            self.barcode = self.barcodes_all
        else:
            self.json_split_path = os.path.join(self.data_path, self.cancer, 'splits', sample_id + '_train_val.json')
            with open(self.json_split_path, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
            self.barcode = json_data[self.mode]

    def transform(self, image):
        image = Image.fromarray(image)
        if self.mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image

    def __len__(self):
        return len(self.barcode)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode[idx]
        image = self.image_dict[barcode]
        image = self.transform(image)
        st_data = self.adate[barcode].to_df().to_numpy().squeeze()
        item['image'] = image.float()
        item['st_data'] = torch.tensor(st_data).float()
        item['barcode'] = barcode
        return item

def build_train_val_loaders(args, cancer, sample_list):
    train_dataset_list = []
    val_dataset_list = []

    for sample in sample_list:
        train_dataset = Path_ST_Dataset(
            data_path=args.data_path,
            cancer=cancer,
            sample_id=sample,
            mode='train'
        )
        train_dataset_list.append(train_dataset)

    for sample in sample_list:
        val_dataset = Path_ST_Dataset(
            data_path=args.data_path,
            cancer=cancer,
            sample_id=sample,
            mode='val'
        )
        val_dataset_list.append(val_dataset)

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
    print('train_len:', len(train_dataset), 'val_len:', len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    print("Finished building loaders")
    return train_loader, val_loader


def calculate_st_embedding(args):
    embedding_dict = {}
    for cancer in args.cancer_list:
        adata_path = os.path.join(args.data_path, str(cancer), 'adata_HVGs')
        adata_path = os.path.join(adata_path, str(os.listdir(adata_path)[0]))
        adata = sc.read_h5ad(adata_path).to_df()
        embedding_length = len(list(adata))
        embedding_dict[cancer] = embedding_length
    return embedding_dict








