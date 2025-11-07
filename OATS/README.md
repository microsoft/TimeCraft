# OATS: Online Data Augmentation for Time Series Foundation Models

This repository contains the code for paper "OATS: Online Data Augmentation for Time Series Foundation Models".

## Overview

OATS introduces a novel online data augmentation framework specifically designed to enhance the training of time series foundation models (TSFM). Unlike traditional offline augmentation methods that pre-generate synthetic data, OATS generates synthetic data by using training samples with high data attribution scores as guiding signals.

OATS consists of three key components:
- Time-series Influence Scores (TSIS) integrate data attribution with time series-specific knowledge to dynamically assess the quality of each training sample, creating a generation guiding signal.
- High-quality Guided Data Augmentation leverages the guiding signal to condition a diffusion model trained on a small subset of the TSFM training data for synthetic data generation.
- Explore-Exploit Mechanism reduces computational overhead and effectively balances between leveraging calculated scores and exploring new samples. The influence scores are stochastically re-evaluated to incorporate model training dynamics ("explore") while preserving previously identified high-quality data ("exploit").


![Method](assets/method.png)

## Environment and Dataset

### Dataset preparation

#### TSFM pretrain dataset
Download dataset for TSFM from [here](https://huggingface.co/datasets/Qingren/TSFM-ScalingLaws-Dataset). The directory organization structure is as follows:

```
- dataset_train
    |- Lotsa16B
    |- Lotsa1B
    |- Lotsa100M
    |- Lotsa10M
- dataset_test
    |- Lotsa16B
    |- Lotsa1B
    |- Lotsa100M
    |- Lotsa10M
    |- LSF
    |- Monash
```

#### Generation model training data
Download extracted dataset from [here](https://huggingface.co/datasets/Theaper/diffusion-training-oats/resolve/main/extracted_diffusion_training_data_LOTSA100M.zip) for diffusion model. The dataset is extracted from the Lotsa100M dataset with a sampling rate 5% of the dataset in 20 selected subdatasets. The directory organization structure is as follows:

```bash
extracted_label_patches_australian_electricity_demand.npy
extracted_label_patches_azure_vm_traces_2017.npy
extracted_label_patches_buildings_900k.npy
extracted_label_patches_CloudOpsTSF_dataset.npy
extracted_label_patches_CMIP6_dataset.npy
...
```

### Environment setting

```bash
# Clone the repository
git clone https://github.com/microsoft/TimeCraft.git
cd TimeCraft/OATS

# Create and activate conda environment
conda env create -f environment.yml
conda activate oats
```


## Quick Start
Step 1. Train a time series generation model with the extracted sampled data.
```bash
cd models/gen_model

python main_train.py --base configs/multi_domain_timedp_local.yaml --gpus 0, --logdir ./logs/ -sl 320 -up -nl 16 --batch_size 128 -lr 0.0001 -s 0
```

Step 2. Train the time series foundation model
```bash
python -m cli.train_val\
       -cp conf/pretrain\
       -cn default_ddp_val_enc\
       model=encoder\
       model.enable_influence_scoring=true\
       data=lotsa100M_weighted\
       val_data=all\
       trainer.logger.project=TSFM_PRETRAIN\
       run_name=encoder10M_etth1_develop\
       model.generate_after_epoch=0\
       model.influence_filter_ratio=1.0\
       model.select_from_generated=false
```

Outputs: The results can be found in wandb log and `./outputs/pretrain/`


