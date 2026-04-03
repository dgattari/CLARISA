
# MARTA trained model
This folder contains the final trained MARTA classifier together with the minimal set of artifacts required to interpret, reuse, and reproduce the model.

The model was trained using a fixed slide-level data split designed to prevent leakage between related ROIs.

---

## Contents

### Model checkpoint
- **`best_stage3_full.pth`**  
  Final trained model checkpoint selected based on minimum validation loss during training.  
  This is the model to be used for inference.

⚠️ The checkpoint file is not stored directly in this repository due to its size.

You can download it from Hugging Face:

👉 https://huggingface.co/jsanchoz/marta-cx43-slide-classifier

After downloading, place the file in this folder: `trained_model/best_stage3_full.pth`

No Hugging Face account is required to download the model.

---

### Training summary
- **`summary.json`**  
  Complete summary of the training run, including:
  - final hyperparameter configuration
  - architecture definition (input mode, fusion, head type)
  - training, validation and test metrics (ROC-AUC, PR-AUC, precision, recall)
  - best validation checkpoint information

This file provides the main reference for interpreting model performance.

---

### Data split definition
- **`split_info.json`**  
  High-level description of the dataset split used during training:
  - split strategy (`manual_groups`)
  - grouping variable (`slide_id`)
  - number of samples in train / validation / test
  - assignment of slides to each partition

This split ensures that ROIs from the same slide are not shared across partitions, reducing data leakage.

---

## Reproducibility artifacts
The `reproducibility/` subfolder contains additional information about how the dataset was partitioned.

### Dataset composition
- **`slide_class_summary.csv`**  
  Per-slide summary of class distribution:
  - number of ROIs per slide
  - class balance (class 0 / class 1)

This highlights dataset heterogeneity across slides.

- **`split_class_summary.csv`**  
  Class distribution across train / validation / test splits:
  - total number of samples
  - proportion of each class per split

Useful to interpret performance metrics, especially under class imbalance.

---

### Sample-level split definition
- **`split_assignments.csv`**  
  Full mapping of each ROI sample:
  - source image
  - slide identifier
  - bounding box coordinates
  - class label
  - assigned split (train / val / test)

This file provides a complete, sample-level traceability of the dataset.

- **`split_indices.json`**  
  Exact indices of samples assigned to each split.

This allows exact reconstruction of the training/validation/test partitions used in all experiments.

---

## Notes on reproducibility
- All experiments (training, hyperparameter tuning and final evaluation) were performed using the same precomputed split.
- Model selection was based on the checkpoint achieving minimum validation loss during training.
- Final performance is reported on the held-out test set defined in `split_info.json`.

---

## Relation to the main repository
For usage of this model (inference, visualization, batch processing), refer to the main repository README:

---

## Summary
This folder is intended to provide:
- a ready-to-use trained model (`.pth`)
- full transparency on how it was trained
- enough metadata to reproduce the exact experimental setup

without requiring access to intermediate training pipelines.