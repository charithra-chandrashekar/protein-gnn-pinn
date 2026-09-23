# Protein PINN-GNN

Physics-informed graph neural network for protein structure modeling.

## Project layout

```
configs/          YAML configs for model, physics, and training
data/             Raw PDBs, processed graphs, and train/val/test splits
src/              Core library (data, physics, models, losses, training)
notebooks/        Exploratory and validation notebooks by phase
tests/            Unit and integration tests
scripts/          CLI entry points
```

## Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 0 | `visualization.py`| Graph sanity checks |
| 1 | `encoder.py`| GNN encoder baseline |
| 2 | `bond_taxonomy.py`, `physics_config.yaml` | Bond taxonomy and geometries |
| 3 | `energy_terms.py`, `energy_module.py` | Differentiable energy function |
| 4 | `physics_loss.py` | Physics regularizer in training |
| 5 | `heads.py` | Task-specific prediction heads |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download a PDB subset
python scripts/download_pdb_subset.py

# Train
python scripts/run_training.py \
  --model-config configs/model_config.yaml \
  --physics-config configs/physics_config.yaml \
  --training-config configs/training_config.yaml
```

## Tests

```bash
pytest tests/
```
