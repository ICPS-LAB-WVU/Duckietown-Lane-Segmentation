# iCPS Duckietown Lane Segmentation

Training and evaluation code for four-class semantic segmentation of Duckietown road/lane markings. The project supports a vanilla U-Net and configurable U-Net variants for lane segmentation task.

## 1. Download the dataset

The image dataset is distributed separately from GitHub as a ZIP file on Google Drive:

- **Dataset ZIP (Google Drive):** https://drive.google.com/file/d/1r2dop_4oLVMfATo5NQ57zTveFTKt9Ed2/view?usp=sharing

Download **`iCPS_Duckietown_Lane_Segmentation_Dataset.zip`**. The ZIP must have these two folders at its top level:

```text
trainTRACKimg/
trainTRACKmask/
```

This release contains 4,950 RGB images and 4,950 corresponding indexed segmentation masks. The four mask IDs are described below.

## 2. Set up the project (Ubuntu / Linux)

```bash
git clone https://github.com/ICPS-LAB-WVU/duckietown-unet-lane-segmentation.git
cd duckietown-unet-lane-segmentation

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p datasets
unzip ~/Downloads/iCPS_Duckietown_Lane_Segmentation_Dataset.zip -d datasets/
```


After extraction the layout **must** be:

```text
duckietown-unet-lane-segmentation/
├── configs/
│   ├── model/
│   └── train/
├── datasets/
│   ├── trainTRACKimg/        # RGB images
│   └── trainTRACKmask/       # indexed PNG masks
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_all.sh
├── src/
├── README.md
├── requirements.txt
└── pyproject.toml
```

Do not create `train/`, `val/` or `test/` subfolders for this dataset. The training pipeline builds its own train/validation split.

## 3. Confirm that the dataset was extracted correctly

From the repository root:

```bash
find datasets/trainTRACKimg -maxdepth 1 -type f | wc -l
find datasets/trainTRACKmask -maxdepth 1 -type f | wc -l
```

Both commands should print **4950**.

To check that the Python environment can import the training package:

```bash
python -c "import torch, duckietown_seg; print('PyTorch:', torch.__version__); print('Import OK')"
```

## 4. Train the baseline U-Net

```bash
python scripts/train.py \
  --model-config configs/model/vanilla.yaml \
  --train-config configs/train/default.yaml
```

The default configuration uses 256 × 256 inputs, batch size 8, AdamW, cross-entropy loss, a maximum of 30 epochs, validation fraction 0.1 and split seed 42. The trainer saves run artifacts under `outputs/`.

For a clean 4,950-pair dataset with the default settings, the expected split is **4,455 training pairs and 495 validation pairs** (557 training batches at batch size 8 if no samples are dropped). If the progress bar shows a different total, check which configuration and dataset root are actually being used.

A first run can take a while on CPU. The command above launches the full experiment; it is not a short smoke test.

## 5. Try another U-Net variant

Swap the model configuration; no code modification is required. For example:

```bash
python scripts/train.py \
  --model-config configs/model/depthwise.yaml \
  --train-config configs/train/default.yaml
```

Available model configurations include `vanilla`, `depthwise`, `se_decoder`, `aspp`, `depthwise_se`, `depthwise_aspp`, `se_aspp` and `full`.

Other loss/training configurations are available in `configs/train/`, including `dice_ce.yaml`, `focal_dice.yaml` and `dsc.yaml`. The `merged_640*.yaml` configurations may use different dataset or input settings; read them before running.

**Instructor note:** `scripts/run_all.sh` launches many model/configuration combinations. Do not use it for a first test.

## 6. Dataset labels and evaluation

Each mask is a single-channel image with categorical pixel IDs:

| ID | Class |
|---|---|
| 0 | Background |
| 1 | Red transverse marking / stop line |
| 2 | White lane marking |
| 3 | Yellow lane marking |

The class labels were identified from the supplied dataset visualizations. The loader uses nearest-neighbor interpolation for resizing segmentation masks so integer labels remain valid.


For evaluating a saved checkpoint, use the repository's `scripts/evaluate.py` command with `--checkpoint` and `--split val`. The checkpoint must be produced by a compatible model configuration.

## 7. Weights & Biases (optional)

Online experiment tracking is **not required** for coursework. In `configs/train/default.yaml` use:

```yaml
logging:
  use_wandb: false
```

To use Weights & Biases for research experiments, change this setting to `true` and configure your own account. Do not commit API keys, `.netrc` files or private experiment credentials.

## 8. Repository contents and citation

The GitHub repository contains the training code, model and experiment configurations, and documentation. The dataset ZIP, generated outputs, W&B run files, environments and model checkpoints are **not** included in Git.

Related publication: M. Al Islam and A. S. El-Wakeel, *Adversarial Robustness Analysis of Cloud-Assisted Autonomous Driving Systems*, IEEE SM 2026. [DOI: 10.1109/SM69703.2026.11614113](https://doi.org/10.1109/SM69703.2026.11614113).

This repository provides training code and data access for education/research. 
