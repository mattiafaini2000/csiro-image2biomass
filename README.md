# CSIRO Image2Biomass

This project predicts pasture biomass from photographs in the [CSIRO Biomass Kaggle challenge](https://www.kaggle.com/competitions/csiro-biomass). Frozen ConvNeXt and DINOv2 representations provide image features; R performs PCA and fits support vector regressors for green, dead and clover dry biomass. All predictions are expressed in grams.

The main workflow is image metadata → frozen image features with image IDs → date-grouped validation and PCA → three RBF SVRs → non-negative component predictions → derived GDM and total biomass → long-format submission.

## Report summary

The [PDF report](Mattia_Faini_eeml_csiro_abstract.pdf), *Predicting Pasture Biomass from Images: A Competition Report on the CSIRO Image2Biomass Kaggle Challenge*, studies an image-based regression task with 357 labelled training photographs. Frozen ConvNeXt and DINOv2 backbones provide representations without fine-tuning. After removing constant features, scaled PCA retains 50 components for three RBF support vector regressors. Negative component predictions are clipped before green dry matter and total dry biomass are derived from their sums.

Spatial tiling improved several reported DINOv2 results. For DINOv2 Large, the public weighted R² rose from 0.65 for whole images to 0.69 with eight tiles, then fell to 0.67 with sixteen tiles. DINOv2 Giant reached 0.70 with eight tiles, both with mean pooling and with additional token statistics. These comparisons support using local image information, while showing that increasing the number of tiles did not produce a steady improvement.

The strongest reported private score was 0.58 for DINOv2 Large with two tiles; the Giant eight-tile variants scored 0.54 and 0.57 privately. Whole-image DINOv2 also scored slightly below ConvNeXt on the private split, despite its higher public score. The report therefore highlights the limits of selecting models from public scores alone and motivates grouping validation images by sampling date. Its four-fold date grouping keeps related observations together; the public/private differences suggest a generalization concern but do not establish its cause.

## Code and inputs

`src/csiro_biomass/` contains Python extraction and cache writing. `R/` contains reusable modelling functions; `scripts/` contains evaluation, prediction and historical tuning entry points. `config/report.R` holds the supplied report's final SVR parameters. Other source parameter sets are recorded in `config/svr_parameter_profiles.csv`. `experiments/` keeps the separate colour-proxy, metadata, boosting and neural-model work.

Supply the competition's `train.csv`, `test.csv`, `sample_submission.csv` and image directories under `data/`, plus a local DINOv2 Hugging Face checkpoint directory or matching Keras ConvNeXt weights under `checkpoints/`. Data and weights are absent and are not downloaded automatically. The [method notes](docs/methods.md) describe the required schemas and available variants.

Python extraction requires NumPy, Pillow, PyTorch and Transformers. ConvNeXt additionally requires Keras 3 with TensorFlow; it does not share DINOv2's PyTorch backend. Dependencies are declared in `pyproject.toml`. The report setup also used pandas; the metadata reader uses Python's CSV library. R evaluation and prediction require `e1071`; historical tuning additionally uses `mlr`, `mlrMBO`, `ParamHelpers` and `DiceKriging`. Optional experiment packages are listed in [r_dependencies.csv](r_dependencies.csv). No historically tested package versions are asserted.

## Example commands

Run from the checkout root with Python and `Rscript` available. These PowerShell examples require the missing inputs and existing dependencies. Set R's startup temporary directory inside the checkout before launching it:

```powershell
New-Item -ItemType Directory -Force .cache/tmp | Out-Null
$env:TMP = $env:TEMP = $env:TMPDIR = (Resolve-Path .cache/tmp).Path
$env:PYTHONPATH = "src"
python -m csiro_biomass.extract --method dinov2-whole --metadata data/train.csv --data-root data --checkpoint checkpoints/dinov2 --model-id local-dinov2 --output features/train
python -m csiro_biomass.extract --method dinov2-whole --metadata data/test.csv --data-root data --checkpoint checkpoints/dinov2 --model-id local-dinov2 --output features/test
Rscript --vanilla scripts/evaluate.R config/report.R
Rscript --vanilla scripts/predict.R config/report.R
```

Use the same extractor, checkpoint and feature schema for training and test images. `CSIRO_DATA_ROOT` can point to an external read-only input root containing `data/`; feature caches remain relative to the checkout's separate `feature_root` configuration.

Whole-image DINOv2 concatenates mean patch tokens followed by the model's pooler output. Tiled extraction supports either that basic mean layout or 17 blocks of token statistics. It requires an explicit row-by-column grid. For example, `--method dinov2-tiled-advanced --grid 1 2` selects a two-tile layout; this example does not specify a historical submission's grid. ConvNeXt requires `--method convnext`, `--convnext-variant large` or `xlarge`, `--pixel-scale raw` or `divide_255`, and `--include-preprocessing true` or `false`, alongside the common input/checkpoint arguments. See the method notes for stage pooling and cropping options.

Each extraction writes `features.csv`, `features.npz`, `manifest.csv` and `schema.json` in its chosen cache directory. Evaluation writes fold assignments, fold scores, a mean-fold and pooled-score summary, identified out-of-fold predictions and support-vector counts to `outputs/evaluation/`. Prediction fits the full training data and writes `pca_svr.rds`, image predictions and `submission.csv` to `outputs/prediction/`. Outputs and library caches stay inside the checkout.

Historical global-PCA tuning is separate from the main evaluation:

```powershell
Rscript --vanilla scripts/tune_svr.R config/report.R Dry_Dead_g
```

It writes to `outputs/tuning_global_pca/Dry_Dead_g/`; it is not the fold-local preprocessing path used for evaluation.

## Historical reported results

These weighted global R² scores are transcribed from the supplied reports. The best public score, 0.70, and best private score, 0.58, belong to different variants.

| Representation | Public | Private |
| --- | ---: | ---: |
| ConvNeXt-XLarge, whole image | 0.58 | 0.54 |
| DINOv2 Large, whole image | 0.65 | 0.53 |
| DINOv2 Large, 2 tiles | 0.66 | 0.58 |
| DINOv2 Large, 4 tiles | 0.68 | 0.57 |
| DINOv2 Large, 8 tiles | 0.69 | 0.56 |
| DINOv2 Large, 16 tiles | 0.67 | 0.55 |
| DINOv2 Giant, 8 tiles | 0.70 | 0.54 |
| DINOv2 Giant, 8 tiles and token statistics | 0.70 | 0.57 |

[reported_scores.csv](results/reported_scores.csv) records attribution. DINOv2's whole-image private score is below ConvNeXt's, so the table does not establish an improvement on every split. Public/private gaps alone do not establish distribution shift.
