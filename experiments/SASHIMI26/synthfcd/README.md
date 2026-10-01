# SynthFCD

These YAML files are configs for the `synthfcd` command in this repository. Each file is one simulation with a single `random_seed`. There is no seed sweep.

```bash
synthfcd path/to/config.yaml
```

The checked-in files use seed 44. The same configs were also run for seed 43: set `random_seed` to `43` and update the seed suffix on `log_file` and `out_dir`.

Each seed's dataset was pretrained separately with anyBrainer (see [training](../training/)), then fine-tuned at three seeds. That yields six fine-tuned models per SynthFCD preset. Reported test-set results are the mean of per-subject predictions across those models; outputs were not ensembled.
