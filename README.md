
# MARTA
Computational pipeline to detect and spatially quantify the lateral redistribution of CX43 in cardiac tissue, generating probability maps and aggregated metrics.

---

> Publication: **link coming soon**

<!-- Add main figure from the paper here -->

---

## Table of contents
- [Installation](#installation)
- [Data structure](#data-structure)
- [Training](#training)
  - [Data split policy](#data-split-policy)
    - [Data split generation on cluster (SLURM)](#data-split-generation-on-cluster-slurm)
  - [Quick start](#quick-start)
  - [Training on cluster (SLURM)](#training-on-cluster-slurm)
  - [Example configuration](#example-configuration)
  - [Training outputs](#training-outputs)
  - [Training configuration](#training-configuration)
  - [Optional expert tracking](#optional-expert-tracking)
  - [Hyperparameter tuning](#hyperparameter-tuning)
- [Inference](#inference)
  - [Single-image inference](#single-image-inference)
  - [Batch inference and grid generation](#batch-inference-and-grid-generation)
  - [Running inference on cluster (SLURM)](#running-inference-on-cluster-slurm)
  - [Inference outputs](#inference-outputs)
  - [Inference configuration](#inference-configuration)
- [Pipeline overview](#pipeline-overview)
- [Contact and support](#contact-and-support)

## Installation
```bash
git clone https://github.com/dgattari/MARTA.git
cd MARTA

conda create -n marta python=3.10
conda activate marta
pip install -r requirements.txt
```

---

## Data structure
The pipeline expects the following structure:

```
data/
├── dataset/   # .spydata annotations
└── images/    # corresponding images (.png / .tif / .tiff)
```

Each `.spydata` file must contain:
* `target_regions_1_filtered` → class 0
* `target_regions_2_filtered` → class 1


<!-- # aqui faltaria meter algo como data availability -->

---

## Training
The training module is intended to learn a classifier that distinguishes ROI patterns associated with longitudinal versus lateralized CX43 distributions in annotated cardiac tissue images.

Starting from `.spydata` annotations and their corresponding images, the pipeline builds ROI-level samples, uses a **precomputed train/validation/test split**, and trains the classifier in three stages.


### Data split policy
MARTA uses a precomputed data split that is created once and then reused consistently in:

- standard classifier training
- hyperparameter tuning
- final model evaluation

This avoids redefining the split at runtime and makes the full training workflow more reproducible and easier to audit.

The split is generated with:

```bash
python -m src.train.create_data_split --config configs/data_split.yaml
```

The resulting split artifacts are stored on disk and include:
- split metadata
- exact train / validation / test indices
- one-row-per-sample split assignments
- class summaries by slide and by final split

Standard training and hyperparameter tuning can then load the same split definition directly from disk.

#### Data split generation on cluster (SLURM)
```bash
sbatch scripts/create_data_split.sh
```

<!-- Hay que discutir si mas adelante lo queremos vender como un metodo ya para usar en inferencia y la parte del training va a otro README o queremos que el usuario entrene con sus propios datos y luego haga al inferencia.  -->

### Quick start  
```bash
python -m src.train.train_classifier --config configs/train_classifier.yaml
```

This will:
* build ROI samples from `.spydata`
* split data into train / validation / test
* train the model in 3 stages (freeze → partial unfreeze → full unfreeze)
* select the best checkpoint by validation loss
* evaluate the selected checkpoint on the test set
* export metrics, logs and ROC curve

### Training on cluster (SLURM)
```bash
sbatch scripts/train_classifier.sh
```
### Example configuration
Recommended default setup:

```yaml
# Recommended default setup
input_mode: stack
fusion: dual
head_kind: mlp
hidden: 128
dropout: 0.5
```
### Training outputs
Outputs are saved in:

```
experiments/marta_classifier/<run_name_timestamp>/
```

Typical outputs include:
- `config.json`
- `split_info.json`
- `best_stage*.pth`
- `train_result.json`
- `summary.json`
- `roc_test.png`
- `metrics_summary.csv`

### Training configuration
All training behavior is controlled via:

```
configs/train_classifier.yaml
```

Main parameters: 

**Data & splits**
* `use_precomputed_split`: whether to load a precomputed split from disk
* `split_dir`: directory containing the saved split artifacts
* `val_size`, `test_size`: only used when a split is generated on the fly for backward compatibility
* `use_group_split`, `group_key`: legacy split options kept for compatibility with alternative split strategies

**Input**
* `input_mode`: 
        - `256`: local ROI context
        - `512`: larger ROI context
        - `stack`: combines both views
* `fusion`: 
        - `dual`: two parallel streams with shared backbone
        - `stack6` (only for `stack`): 6-channel input

**Training**
* `stage1_epochs`, `stage2_epochs`, `stage3_epochs`: three training phases
* `k_unf`: number of backbone blocks unfrozen in stage 2

**Optimization**
* `head_lr`, `last_lr`, `rest_lr`: learning rates for different parts of the model
* `weight_decay`: L2 regularization
* `class1_bonus`: positive-class weighting factor
* `decision_threshold`: threshold used for label assignment during evaluation

**Model**
* `head_kind`: `mlp` | `logreg`
* `hidden`: hidden layer size for MLP heaad
* `dropout`: dropout probability for MLP head

### Optional expert tracking
Advanced experiment tracking is available through an optional expert mode for users who want W&B monitoring, artifact logging and easier comparison across runs.

See:
`docs/EXPERT_TRACKING.md`

### Hyperparameter tuning
A dedicated hyperparameter tuning workflow is available for controlled architecture screening and fine-tuning optimization of the MARTA classifier.

Hyperparameter tuning reuses the same precomputed split used by standard training, so all trials are evaluated on exactly the same train / validation partitions.

For methodology, configuration, outputs, and execution examples, see:
`docs/HYPERPARAMETER_TUNING.md`

---


## Inference
The inference module is intended to apply a trained MARTA checkpoint to new cardiac tissue images in order to detect ROIs, classify them, and summarize the spatial distribution of CX43 across the image or across multiple sections.

Given one or more input images and a trained checkpoint, the pipeline performs ROI detection, ROI-by-ROI inference, embedding analysis, heatmap generation, and export of quantitative and visual outputs. 

### Single-image inference
```bash
python -m src.inference.single_image \
    --image path/to/image.png \
    --ckpt path/to/best_stage3_full.pth \
    --outdir infer_test \
    --config configs/inference.yaml
```

This will:
- read the input image
- detect candidate ROIs
- load the trained checkpoint
- run ROI-by-ROI inference
- save ROI tables
- generate classification overlays
- generate interactive HTML visualization
- compute t-SNE on ROI embeddings
- generate lateralization heatmaps
- save a final summary.json

### Batch inference and grid generation
```bash
python -m src.inference.batch_grid \
    --folder_images path/to/folder \
    --ckpt path/to/best_stage3_full.pth \
    --outdir infer_batch \
    --config configs/inference.yaml
```

This will:
- discover and sort section images
- run single-image inference for each section
- collect the final overlays
- assemble a combined grid image across sections

### Running inference on cluster (SLURM)
```bash
sbatch scripts/inference_batch.sh
```

### Inference outputs
Typical outputs for single-image inference include:
- `*_resultados.csv`
- `*_resultados.xlsx`
- `*_clasificacion_coloreada.jpg`
- `*_resultados.html`
- `*_tsne.csv`
- `*_tsne.html`
- `*_heatmap_arealateralizacion.jpg`
- `*_lcr_overlay_withbar.jpg`
- `summary.json`

For batch inference, outputs include one subfolder per section plus:
`combined_heatmaps_grid.jpg`

### Inference configuration
All inference behavior is controlled via:
```text
configs/inference.yaml
```

Suggested default configuration:
```yaml
resize_to: 384

threshold: 0.50
expand: 40

perplexity: 30.0
soft: false
sigma: 128.0

random_seed: 42
save_excel: true
```

Main parameters:
- `resize_to`: final spatial size used before model inference
- `threshold`: minimum confidence required to assign class 0 or class 1
- `expand`: number of pixels used to expand ROI candidates before recentering
- `perplexity`: t-SNE perplexity
- `soft`: enables soft gaussian-smoothed heatmap
- `sigma`: gaussian smoothing parameter for soft heatmap
- `save_excel`: whether to export .xlsx results in addition to .csv

---

### Pipeline overview

#### Training
```text
.spydata + images
        ↓
ROI extraction
        ↓
precomputed train / val / test split
        ↓
3-stage training
        ↓
best checkpoint selection
        ↓
test evaluation + ROC
```

#### Inference
```text
image(s)
   ↓
ROI detection
   ↓
checkpoint loading
   ↓
ROI-by-ROI inference
   ↓
tables + overlays + t-SNE + heatmap
   ↓
final metrics + summary
```

---

Note for developers: the original version of code is still in `MARTA/old_code` <!-- eliminar luego esto -->

## Contact and support
For questions about the method, codebase, or usage of MARTA, please contact:

- Daniel Eduardo Gattari - DGattari@austral.edu.ar
- Joseba Sancho-Zamora — jsanchoz@unav.es