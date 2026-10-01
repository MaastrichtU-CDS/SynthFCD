# SASHIMI 2026

These configs reproduce the findings of the SASHIMI 2026 publication (paper link to be added). The following steps were applied:

1. Data curation and skull-stripping for the pretraining and real FCD cohorts ([preprocessing](preprocessing/), step 0).
2. Structural MRI preprocessing for both cohorts ([preprocessing](preprocessing/), steps 1–5).
3. Anatomical segmentation masks ([synthseg](synthseg/)).
4. Synthetic FCD simulation for different presets ([synthfcd](synthfcd/)).
5. Pretraining on different presets and the baseline ([training](training/)).
6. Fine-tuning and evaluation on the real FCD cohort ([training](training/)).
