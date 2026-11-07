<div align="center">

# MOBA-Net

### Spatial Prior Guided Mixture-of-Boltzmann-Attention Network for Lunar Landform Recognition

[![Framework](https://img.shields.io/badge/Framework-PyTorch%202.5.1-ee4c2c.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Detectron2%200.6-459bd8.svg)](https://github.com/facebookresearch/detectron2)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](#license)

</div>

Official implementation of **MOBA-Net**, a unified multi-task network for **target detection**, **instance segmentation**, and **semantic segmentation** of lunar landforms (craters and lineaments) on digital orthophoto map (DOM) data.

## 🚀 Framework

MOBA-Net adopts [MaskDINO](https://github.com/IDEA-Research/MaskDINO) as the baseline and consists of a ResNet-50 backbone, four **DST encoders** (deformable switch transformer), four **MOBA decoders**, and the **CDIF block**. Each MOBA decoder is composed of an **SP-MOBA module** and a **CAFFN**.

```
DOM image ──► ResNet-50 ──► 4 × DST Encoders ──► 4 × MOBA Decoders ──► CDIF ──► Det. + Ins. + Sem.
                                                │                        │
                                                ├─ SP-MOBA (SP-BAEs)     ├─ train: adaptive inference fusion
                                                └─ CAFFN (DPG + LREs)    └─ test : hybrid inference-query selection
```

| Module | Class / config | Description |
| :--- | :--- | :--- |
| SP-MOBA module | `SPMOBA` (`ATTN_TYPE_DEC: "sp_moba"`) | Gate + 4 SP-BAEs with top-2 routing; Gaussian prior sampling on Boltzmann probability fields |
| CAFFN | `CAFFN` (`FFN_TYPE_DEC: "caffn"`) | DPG (4 statistics) + 4 LREs with compression ratios r = 4/8/16/32, top-2 routing |
| CDIF block | `cdif_fusion_weight` in `MOBA_Decoder` | Softmax-weighted fusion of decoder inferences (training); concatenated inferences (testing) |
| DST encoders | `DST_Encoder` / `DST` (`ATTN_TYPE_ENC: "dst"`) | Deformable transformer blocks with 4 attention experts and 4 FFN experts, top-2 routing |

## 🛠️ Installation

**Requirements** (verified environment):

| Package | Version |
| :--- | :--- |
| Python | 3.12 |
| PyTorch + CUDA | 2.5.1 + cu124 |
| torchvision | 0.20.1 |
| detectron2 | 0.6 |
| triton | 3.1.0 |
| opencv-python-headless | ≥ 4.12 |

```bash
# 1. Create the conda environment
conda create -n moba_net python=3.12 -y
conda activate moba_net

# 2. Install PyTorch (cu124) and detectron2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.5/index.html

# 3. Clone this repository
git clone https://github.com/ningyang-li/MOBA-Net.git
cd MOBA-Net

# 4. Compile the CUDA operations (MultiScaleDeformableAttention & ParallelExperts)
cd moba_net/modeling/pixel_decoder/ops
bash make.sh
cd ../../../modeling/moe/base
pip install -e .
cd ../../../../
```

> 💡 If the deformable-attention CUDA op is already available in your environment (e.g. installed via `pip install MultiScaleDeformableAttention`), step 4 can be skipped.

## 📦 Datasets and Preparation

MOBA-Net is evaluated on three public lunar landform datasets: ChangE, LU, and LRO-L4.
Organize each dataset in COCO format under `datasets/` with three splits (`train2017`, `val2017`, and `test2017`; the semantic ground truth used by the evaluator is stored in `annotations/sem/`):

```
MOBA-Net/
└── datasets/
    └── <DATASET>/                  # ChangE | LU | LRO-L4
        ├── train2017/              # training DOM images (.png)
        ├── val2017/                # validation DOM images (.png)
        ├── test2017/               # testing DOM images (.png)
        └── annotations/
            ├── instances_sem_train2017.json   # COCO instance annotations
            ├── instances_sem_val2017.json
            ├── instances_sem_test2017.json
            └── sem/
                ├── train2017/      # semantic segmentation ground truth
                ├── val2017/
                └── test2017/
```

The dataset is registered automatically by [`train_net.py`](train_net.py). Select the dataset to train/evaluate with the `DATASET` variable at the top of the script:

```python
# train_net.py
DATASET = 'ChangE'   # ChangE | LU | LRO-L4
```

## 🔀 Data Split and Evaluation Protocol

Following the protocol in our paper, the original training set of each dataset is split into a training set and a validation set at a ratio of about 9:1, and the test set is kept untouched:

| Dataset | Train | Validation | Test |
| :--- | :---: | :---: | :---: |
| ChangE | 595 | 66 | 197 |
| LU | 7416 | 824 | 1031 |
| LRO-L4 | 1168 | 132 | 380 |

**Split protocol.** Parameter-setting and ablation experiments are trained on the training set and evaluated on the validation set. The final model and all compared methods are trained on the combination of the training and validation sets and evaluated on the test set. The split manifests are defined by the image lists in the three annotation files of each dataset (`instances_sem_train2017.json`, `instances_sem_val2017.json`, and `instances_sem_test2017.json`).

**Random seeds.** All experiments are repeated five times under five fixed seeds (42, 43, 44, 45, and 46) shared by all methods, and the average results and standard deviations of the five runs are reported. The only source of randomness across the five runs is the Xavier initialization of the network parameters, which varies with the random seed; the data ordering, optimizer settings, and data split are kept fixed.

**Best-checkpoint selection.** The best checkpoint (`model_best.pth`) is selected based on the highest instance-segmentation AP50 (Ins-AP50) on the validation set.

## 🏋️ Training

Download the ImageNet-pretrained ResNet-50 weights (configured via `detectron2://ImageNetPretrained/torchvision/R-50.pkl`, fetched automatically), then run (our training card is NVIDIA RTX 5880 Ada Generation 48GB):

```bash
# ChangE
python train_net.py --num-gpus 1 --config-file configs/moba_R50_ChangE.yaml

# LU
python train_net.py --num-gpus 1 --config-file configs/moba_R50_LU.yaml

# LRO-L4
python train_net.py --num-gpus 1 --config-file configs/moba_R50_LRO-L4.yaml
```

To resume training or fine-tune from a checkpoint:

```bash
python train_net.py --num-gpus 1 --config-file configs/moba_R50_ChangE.yaml \
    MODEL.WEIGHTS output/model_best.pth
```

## ⬇️ Pretrained Weights

The pretrained MOBA-Net weights for the three datasets (the final models trained on the combination of the training and validation sets; see the protocol above) can be downloaded via [Google Drive](https://drive.google.com/drive/folders/1mITcosw3MzLahxabi6Dt1UfM-alA0BOD). Place the downloaded checkpoint under `output/` (e.g. `output/model_best.pth`), and run the evaluation or visualization scripts below.

## 📈 Evaluation

```bash
python train_net.py --eval-only --num-gpus 1 \
    --config-file configs/moba_R50_ChangE.yaml \
    MODEL.WEIGHTS output/model_best.pth OUTPUT_DIR output_vis
```

## 🔮 Visualization

[`predict.py`](predict.py) runs inference on the test set and saves the predicted instance masks, bounding boxes, semantic maps, and the ground-truth comparisons:

```bash
python predict.py   # set DATASET and config_file inside the script first
```

Results are written to `vis/`, including `*_pred_instance_no_text.png`, `*_pred_bbox.png`, `*_pred_semantic.png`, and the corresponding ground-truth overlays.

## 🙏 Acknowledgements

This project is built upon [MaskDINO](https://github.com/IDEA-Research/MaskDINO), [Detectron2](https://github.com/facebookresearch/detectron2), and the Boltzmann attention sampling of [BoltzFormer](https://aka.ms/boltzformer). We thank the authors for their excellent work. This work was supported by the National Key Research and Development Program of China under Grant 2023YFB3906102.

## ⚖️ License

This repository is released under the [Apache 2.0 license](LICENSE), following its MaskDINO and Detectron2 lineage.

## 💬 Discussion

Feedback and discussions are warmly welcome! 👋 If you have any questions ❓, suggestions, or ideas 💡 about this project. We are always glad to exchange ideas 😊 and will respond as soon as possible 🚀.
