# SynthSeg

Anatomical masks for the pretraining cohort were generated with [synthseg-wrapper](https://github.com/pkoutsouvelis/cli-wrappers/tree/main/tools/synthseg) (`tools/synthseg` in [pkoutsouvelis/cli-wrappers](https://github.com/pkoutsouvelis/cli-wrappers)). The wrapper runs [SynthSeg](https://github.com/BBillot/SynthSeg) (Billot et al., [Medical Image Analysis, 2023](https://doi.org/10.1016/j.media.2023.102789)). Input selection uses the same criteria as HD-BET skull-stripping: healthy subjects with both a T1w and a FLAIR scan. Only T1w is segmented; `synthseg_t1.yaml` is a placeholder for that run.

```bash
synthseg-wrapper -c path/to/config.yaml
```
