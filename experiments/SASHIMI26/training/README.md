# Training

These YAML files are configs for [anyBrainer](https://github.com/MaastrichtU-CDS/anyBrainer). Run each one with `SweepWorkflow`:

```bash
anyBrainer SweepWorkflow path/to/config.yaml
```

Each config sweeps training seeds, then evaluates the trained model on the test set. No hyperparameter tuning was performed.
