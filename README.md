# CLARISA
Computational pipeline to detect and spatially quantify the lateral redistribution of CX43 in cardiac tissue, generating probability maps, classification overlays, and global lateralization metrics.

---

> Publication: **link coming soon**

---

<p align="center">
  <img src="docs/Figure_2.png" alt="Figure 2. CLARISA method overview" width="100%">
</p>

*Figure 2. Overview of the CLARISA method.*

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
  - [Using a pretrained model](#using-a-pretrained-model)
  - [Single-image inference](#single-image-inference)
  - [Batch inference and grid generation](#batch-inference-and-grid-generation)
  - [Running inference on cluster (SLURM)](#running-inference-on-cluster-slurm)
  - [Inference outputs](#inference-outputs)
  - [Inference configuration](#inference-configuration)
- [Expert annotation and comparison tools](#expert-annotation-and-comparison-tools)
  - [Overview](#overview)
  - [Interactive expert annotation](#interactive-expert-annotation)
  - [Computing outputs from expert annotations](#computing-outputs-from-expert-annotations)
  - [Notes and reproducibility](#notes-and-reproducibility)
- [Pipeline overview](#pipeline-overview)
- [Contact and support](#contact-and-support)

## Installation
```bash
git clone https://github.com/dgattari/CLARISA.git
cd CLARISA

conda create -n clarisa python=3.10
conda activate clarisa
pip install -r requirements.txt
```

For the expert annotation workflow, Flask is also required:

```bash
pip install flask
```

---

## Data structure
The pipeline expects the following structure:

```text
data/
├── dataset/   # .spydata annotations
└── images/    # corresponding images (.png / .tif / .tiff)
```

Each `.spydata` file must contain:
- `target_regions_1_filtered` → class 0
- `target_regions_2_filtered` → class 1

---

## Training
The training module is intended to learn a classifier that distinguishes ROI patterns associated with terminal versus lateralized CX43 distributions in annotated cardiac tissue images.

Starting from `.spydata` annotations and their corresponding images, the pipeline builds ROI-level samples, uses a **precomputed train/validation/test split**, and trains the classifier in three stages.

### Data split policy
CLARISA uses a precomputed data split that is created once and then reused consistently in:

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

### Quick start
```bash
python -m src.train.train_classifier --config configs/train_classifier.yaml
```

This will:
- build ROI samples from `.spydata`
- split data into train / validation / test
- train the model in 3 stages (freeze → partial unfreeze → full unfreeze)
- select the best checkpoint by validation loss
- evaluate the selected checkpoint on the test set
- export metrics, logs and ROC curve

### Training on cluster (SLURM)
```bash
sbatch scripts/train_classifier.sh
```

### Example configuration
Recommended default setup:

```yaml
input_mode: stack
fusion: dual
head_kind: mlp
hidden: 128
dropout: 0.5
```

### Training outputs
Outputs are saved in:

```text
experiments/clarisa_classifier/<run_name_timestamp>/
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

```text
configs/train_classifier.yaml
```

Main parameters:

**Data & splits**
- `use_precomputed_split`: whether to load a precomputed split from disk
- `split_dir`: directory containing the saved split artifacts
- `val_size`, `test_size`: only used when a split is generated on the fly for backward compatibility
- `use_group_split`, `group_key`: legacy split options kept for compatibility with alternative split strategies

**Input**
- `input_mode`
  - `256`: local ROI context
  - `512`: larger ROI context
  - `stack`: combines both views
- `fusion`
  - `dual`: two parallel streams with shared backbone
  - `stack6` (only for `stack`): 6-channel input

**Training**
- `stage1_epochs`, `stage2_epochs`, `stage3_epochs`: three training phases
- `k_unf`: number of backbone blocks unfrozen in stage 2

**Optimization**
- `head_lr`, `last_lr`, `rest_lr`: learning rates for different parts of the model
- `weight_decay`: L2 regularization
- `class1_bonus`: positive-class weighting factor
- `decision_threshold`: threshold used for label assignment during evaluation

**Model**
- `head_kind`: `mlp` | `logreg`
- `hidden`: hidden layer size for MLP head
- `dropout`: dropout probability for MLP head

### Optional expert tracking
Advanced experiment tracking is available through an optional expert mode for users who want W&B monitoring, artifact logging and easier comparison across runs.

See:
`docs/EXPERT_TRACKING.md`

### Hyperparameter tuning
A dedicated hyperparameter tuning workflow is available for controlled architecture screening and fine-tuning optimization of the CLARISA classifier.

Hyperparameter tuning reuses the same precomputed split used by standard training, so all trials are evaluated on exactly the same train / validation partitions.

For methodology, configuration, outputs, and execution examples, see:
`docs/HYPERPARAMETER_TUNING.md`

---

## Inference
The inference module applies a trained CLARISA checkpoint to new cardiac tissue images to detect ROIs, classify them, and summarize the spatial distribution of CX43 across the image.

For each processed image, the pipeline produces four unified outputs:

1. A **continuous lateralization heatmap** built by spatial interpolation of ROI-level probabilities.
2. A **classification overlay** where each detected CX43-positive region is filled with a color indicating its predicted class (terminal, lateralized, or indeterminate).
3. **Area-based global metrics** (primary):
   - `pct_lat_area_all`: percentage of CX43-positive area classified as lateralized, over the total detected CX43 area.
   - `pct_lat_area_conf`: same, but restricted to ROIs with a confident label (excluding indeterminate).
4. **Heatmap-based global metrics** (complementary):
   - `pct_lat_heat_all`, `pct_lat_heat_conf`: mean of the continuous heatmap over the same areas.

### Using a pretrained model
This repository provides a ready-to-use trained model via the `/trained_model/` folder.

Due to file size limitations, the model checkpoint must be downloaded separately from Hugging Face:

👉 https://huggingface.co/jsanchoz/clarisa-cx43-slide-classifier

After downloading, place the checkpoint file at:

```bash
trained_model/best_stage3_full.pth
```

This folder contains:
- a final trained checkpoint (`best_stage3_full.pth`)
- training summary and performance metrics
- the exact data split used during training
- additional reproducibility artifacts

You can directly use this model for inference without retraining:

```bash
--ckpt trained_model/best_stage3_full.pth
```

Alternatively, you may train your own model and use your own checkpoint.

For full details about the provided model and its provenance, see:
`trained_model/README_model.md`

### Single-image inference
```bash
python -m src.inference.single_image \
    --image path/to/image.png \
    --ckpt trained_model/best_stage3_full.pth \
    --outdir infer_test \
    --config configs/inference.yaml
```

This will:
- read the input image
- detect candidate ROIs
- load the trained checkpoint
- run ROI-by-ROI inference
- save ROI tables (CSV / XLSX)
- generate a classification overlay
- generate an interactive HTML visualization
- generate the continuous lateralization heatmap and its overlay on the original image
- compute the four global lateralization metrics
- save a final `summary.json`

### Batch inference and grid generation
```bash
python -m src.inference.batch_grid \
    --folder_images path/to/folder \
    --ckpt trained_model/best_stage3_full.pth \
    --outdir infer_batch \
    --config configs/inference.yaml
```

This will:
- discover and sort section images
- run single-image inference for each section
- collect the heatmap overlays
- assemble a combined grid image across sections

### Running inference on cluster (SLURM)
```bash
sbatch scripts/inference_batch.sh
```

### Inference outputs
Typical outputs for single-image inference include:
- `*_resultados.csv` / `*_resultados.xlsx` — per-ROI predictions
- `*_clasificacion_coloreada.jpg` — classification overlay
- `*_resultados.html` — interactive visualization of ROIs over the image
- `*_heatmap.jpg` — continuous lateralization heatmap (grayscale with colorbar)
- `*_heatmap_overlay.jpg` — heatmap blended with the original image
- `summary.json` — global metrics and paths to all artifacts

For batch inference, outputs include one subfolder per section plus:
`combined_heatmaps_grid.jpg`

### Inference configuration
All inference behavior is controlled via:

```text
configs/inference.yaml
```

The YAML exposes the key parameters that were previously hardcoded in the inference workflow and can now be modified directly from the configuration file.

**General inference**
- `resize_to` — final spatial size used before model inference
- `threshold` — minimum confidence required to assign class 0 or class 1
- `save_excel` — whether to export `.xlsx` results in addition to `.csv`

**ROI detection (thresholding + morphology)**
- `thresh_value` — intensity threshold for binarization
- `kernel_open` — structuring element size for morphological opening (`1` disables the operation)
- `kernel_dilate` — structuring element size for dilation (`1` disables the operation)
- `expand` — bounding-box expansion in pixels before centered crop extraction

**Crop sizes**
- `crop_local`
- `crop_context`

These control the window sizes extracted around each detected ROI before classification. They should be scaled proportionally to the imaging resolution of the target image so that the physical field of view seen by the model is preserved.

**Heatmap interpolation**
- `sigma` — standard deviation of the Gaussian kernel used to build the continuous lateralization map

The YAML file contains inline documentation explaining the role of each parameter and how to adapt them when applying the method to images acquired under different conditions.

---

## Expert annotation and comparison tools

### Overview
The repository includes auxiliary tools that support qualitative comparison and tissue-level metric computation from expert annotations.

Two scripts are provided under:

```text
src/annotation/
├── expert_annotation_tool.py
└── compute_from_expert_csv.py
```

These tools are designed to remain fully consistent with the inference pipeline:

- they read the same `configs/inference.yaml`
- they use the same ROI detection parameters (`thresh_value`, `kernel_open`, `kernel_dilate`, `expand`)
- they use the same crop sizes (`crop_local`, `crop_context`)
- they reuse the same downstream heatmap and global-metric computation functions

This ensures that the expert annotates exactly the same automatically detected ROIs that are later used for comparison.

### Interactive expert annotation
```bash
python -m src.annotation.expert_annotation_tool \
    --image path/to/image.tif \
    --outdir expert_annotations \
    --config configs/inference.yaml
```

This launches a local web server with an interactive interface that:

- detects ROIs using the same pipeline used at inference time
- allows the expert to navigate the image with zoom and pan
- displays local and contextual crops for each ROI
- allows the expert to label each ROI as:
  - terminal (`0`)
  - lateralized (`1`)
  - uncertain (`-1`)
- saves annotations progressively to a CSV file

By default, the server runs on:

```text
http://localhost:5000
```

Optional arguments:

```bash
--port 5000
--resume
```

Use `--resume` to continue a previous annotation session from an existing CSV.

### Computing outputs from expert annotations
```bash
python -m src.annotation.compute_from_expert_csv \
    --image path/to/image.tif \
    --csv expert_annotations/<image>_expert_annotations.csv \
    --outdir expert_results \
    --config configs/inference.yaml
```

This script:
- re-detects the ROIs using the same inference configuration used during annotation
- reconstructs a ROI-level results structure compatible with the standard inference pipeline
- generates a classification overlay
- generates the continuous heatmap and its overlay on the original image
- computes the same global lateralization metrics used for model inference
- saves a final `summary.json`

Internally, expert labels are converted to probabilities as follows:
- class `1` → `prob_1 = 1.0`
- class `0` → `prob_1 = 0.0`
- class `-1` → `prob_1 = 0.5`

This convention allows expert annotations to be processed with the same interpolation and tissue-level metric functions used for the model outputs.

### Notes and reproducibility
- The same image and the same `configs/inference.yaml` should be used during annotation and post-processing.
- If the YAML parameters are changed between annotation and post-processing, ROI detection may change and ROI indices may no longer match the original expert CSV.
- In the current implementation, the proposed method may produce no indeterminate ROIs when the default threshold `τ = 0.5` is used, in which case `%Lat_area_all` and `%Lat_area_conf` coincide.
- The expert workflow is intended both for qualitative visualization and for tissue-level comparison between the classifier and human annotation.

---

## Pipeline overview

### Training
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

### Inference
```text
image(s)
   ↓
ROI detection
   ↓
checkpoint loading
   ↓
ROI-by-ROI inference
   ↓
heatmap + classification overlay
   ↓
four global metrics + summary
```

### Expert annotation and comparison
```text
image
  ↓
ROI detection from configs/inference.yaml
  ↓
interactive expert labeling
  ↓
expert_annotations.csv
  ↓
expert heatmap + overlay + metrics
```


## Contact and support
For questions about the method, codebase, or usage of CLARISA, please contact:

- Daniel Eduardo Gattari — DGattari@austral.edu.ar
- Joseba Sancho-Zamora — jsanchoz@unav.es
