
# Expert tracking
This project supports an optional expert mode for experiment tracking with Weights & Biases (W&B).

If expert mode is disabled, training runs normally and only local outputs are generated.

## Enable expert mode
In `configs/train_classifier.yaml`:

```yaml
expert_mode: true
expert_config_path: "configs/expert/wandb.yaml"
```

## Local setup
If `expert_mode: true`, the user needs two things.

1. **Install W&B**
```bash
pip install wandb
```

2. **Authenticate**
Quick option:
```bash
wandb login
```

W&B will ask for the API key and store the session locally.

3. **Run training**
python -m src.train.train_classifier --config configs/train_classifier.yaml

---

## Local setup without `wandb login`
A secret environment file can also be used.

Create:

`scripts/secrets/wandb.env`

export WANDB_API_KEY="tu_api_key"
export WANDB_ENTITY="tu_usuario_o_team"

Load it before running:
```bash
source scripts/secrets/wandb.env
python -m src.train.train_classifier --config configs/train_classifier.yaml
```

---

## SLURM setup
Same idea.

1. **Create secret file**
`scripts/secrets/wandb.env`
```bash
export WANDB_API_KEY="tu_api_key"
export WANDB_ENTITY="tu_usuario_o_team"
```

2. **Load it in the SLURM script**
Add this before the `python -m ...` call:

```bash
if [ -f scripts/secrets/wandb.env ]; then
    source scripts/secrets/wandb.env
fi
```

3. **Enable expert mode**
In `configs/train_classifier.yaml`:

```yaml
expert_mode: true
expert_config_path: "configs/expert/wandb.yaml"
```

4. **Submit**
```bash
sbatch scripts/train_classifier.sh
```

---

## Example W&B config
`configs/expert/wandb.yaml`

```yaml
wandb:
  enabled: true
  project: "MARTA-training"
  entity: null
  mode: "online"
  tags: ["marta", "classifier"]
  group: null
  job_type: "train"
  name: null
  notes: null

  watch:
    enabled: true
    log: "all"
    log_freq: 100

  log_artifacts: true
  save_model_artifact: true

  batch_logging:
    enabled: false
    log_every_steps: 20
```

---

## Notes
- If `expert_mode: false`, nothing related to W&B is required.
- `WANDB_ENTITY` can be omitted if the local W&B user/session already defines it.
- Secret files should not be committed to git.
