# CMRCNet
1. Download datasets from https://huggingface.co/datasets/MahmoodLab/hest.  
Please refer to the instructions in [HEST-1k](https://github.com/mahmoodlab/HEST) to download the HEST-1k benchmark datasets.  
Or you can download this data manually from the link: https://huggingface.co/datasets/MahmoodLab/hest  
We provide sample IDs for the six datasets we used:  
COAD: TENX111, TENX147, TENX148, TENX149  
IDC: NCBI783, NCBI785, TENX95, TENX99  
LYMPH_IDC: NCBI681, NCBI682, NCBI683, NCBI684  
READ: ZEN36, ZEN40, ZEN48, ZEN49  
SKCM: TENX115, TENX117  
PSC: NCBI672, NCBI673, NCBI674, NCBI675  

##Dataset
```
CMRCNet/
├── data/
│   ├── COAD/
        ├──patches/
        ├──adata/
│   └── READ/

```
3. Prepare spatial transcriptomics data: python utils_data/pre_process.py
4. Split training sets and test sets: python utils_data/splits.py
  
 ##Dataset_prepared
```
CMRCNet/
├── data/
│   ├── COAD/
        ├──patches/
        ├──adata/
        ├──adata_HVGs/
        ├──splits/
│   └── READ/

```  
6. train: python train.py
7. inference: python inference.py

This work is based on [BLEEP](https://github.com/bowang-lab/BLEEP), [MCFN](https://github.com/dingsaisai/MCFN) and [HEST-1k](https://github.com/mahmoodlab/HEST).
