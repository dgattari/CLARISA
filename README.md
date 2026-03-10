# MARTA
Computational pipeline to detect and spatially quantify the lateral redistribution of CX43 in cardiac tissue, generating probability maps and aggregated metrics that can be associated with ischemic pathology.

## Installation
Clone the repository and create a Conda environment.

```bash
git clone https://github.com/dgattari/MARTA.git
cd MARTA

conda create -n marta python=3.10
conda activate marta
pip install -r requirements.txt
```
---

## Usage
### Training module
Run the training script:

```bash
python MARTA/MARTA_MULTIINPUT_SINGLE_TRAIN.py
```

---

### Inference module
#### Single image
```bash
python MARTA_INFER_TSNE_MULTIINPUT_AREALAT_v2.py \
 --image /path/to/image.png \
 --ckpt /path/to/best_stage3_full.pth \
 --outdir /path/to/output_folder
```

#### Batch inference (multiple images)
```bash
python MARTA_INFER_BATCH_GRID.py \
 --folder_images /path/to/images_folder \
 --ckpt /path/to/best_stage3_full.pth \
 --outdir /path/to/output_folder
```

<!-- Modulo de inferencia
---------------------
Caso simple (1 imagen)
Ejecutas directamente:

python MARTA_INFER_TSNE_MULTIINPUT_AREALAT_v2.py \
 --image /scratch/jsanchoz/MARTA/images/IM133.png \
 --ckpt "/scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE/multi_stack_dual_20260307_202622/mlp_128_d0.5/best_stage3_full.pth" \
 --outdir "/scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE/multi_stack_dual_20260307_202622/mlp_128_d0.5/inference"

Caso simple (1 imagen)
Ejecutas directamente:

python MARTA_INFER_BATCH_GRID.py \
 --folder_images /scratch/jsanchoz/MARTA/images \
 --ckpt "/scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE/multi_stack_dual_20260307_202622/mlp_128_d0.5/best_stage3_full.pth" \
 --outdir "/scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE/multi_stack_dual_20260307_202622/mlp_128_d0.5/inference_batch_grid"
 -->
