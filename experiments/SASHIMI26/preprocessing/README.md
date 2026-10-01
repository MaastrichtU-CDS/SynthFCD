# Preprocessing

Step 0 skull-strips each cohort with [hd-bet-wrapper](https://github.com/pkoutsouvelis/cli-wrappers/tree/main/tools/hdbet) (`tools/hdbet` in [pkoutsouvelis/cli-wrappers](https://github.com/pkoutsouvelis/cli-wrappers)). The wrapper runs [HD-BET](https://github.com/MIC-DKFZ/HD-BET) (Isensee et al., [Human Brain Mapping, 2019](https://doi.org/10.1002/hbm.24750)). The `[synthfcd]` configs also keep healthy T1w–FLAIR pairs from FOMO300k. Placeholders, one T1w and one FLAIR run per cohort:

- `step0[synthfcd]-hdbet_t1.yaml`, `step0[synthfcd]-hdbet_flair.yaml`
- `step0[bonnfcd]-hdbet_t1.yaml`, `step0[bonnfcd]-hdbet_flair.yaml`

```bash
hd-bet-wrapper -c path/to/config.yaml
```

From step 1 onward, the YAML files are configs for [niiflow-preproc](https://github.com/pkoutsouvelis/niiflow/tree/main/packages/niiflow-preproc) 0.3.0 (package in [pkoutsouvelis/niiflow](https://github.com/pkoutsouvelis/niiflow)). Run each file with the `dynamic_workflow` command:

```bash
niiflow-preproc dynamic_workflow path/to/config.yaml
```

Run the numbered steps in order. Bracketed names (`[bonnfcd]`, `[synthfcd]`, …) mark which cohort the step applies to (`synthfcd` for pretraining, `bonnfcd` for fine-tuning).
