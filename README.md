# CMRCNet
1. Download datasets from https://huggingface.co/datasets/MahmoodLab/hest
2. Prepare spatial transcriptomics data: python utils_data/pre_process.py
3. Split training sets and test sets: python utils_data/splits.py
4. train: python train.py
5. inference: python inference.py

This work is based on [BLEEP](https://github.com/bowang-lab/BLEEP) and [MCFN](https://github.com/dingsaisai/MCFN)
