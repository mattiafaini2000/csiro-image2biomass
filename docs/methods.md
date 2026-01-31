# Method notes

**Targets and scoring**

The five targets are `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g` and `Dry_Total_g`, in that order, with weights 0.1, 0.1, 0.1, 0.2 and 0.5. Targets are dry biomass in grams. Three epsilon-SVRs predict green, dead and clover components. Each component is clipped at zero before deriving `GDM_g = green + clover` and `Dry_Total_g = green + dead + clover`.

`R/metrics.R` computes one weighted global R² across every image/target pair: `1 - sum(w * (y - prediction)^2) / sum(w * (y - weighted_global_mean)^2)`. The denominator uses one mean, `sum(w * y) / sum(w)`, rather than five separate target means. The mean of four fold scores and the score of pooled out-of-fold predictions are reported separately.

**Metadata, identities and interchange**

Training metadata is long-format CSV with `image_path`, `Sampling_Date`, `target_name` and numeric `target`; each image must have all five targets and one date. Test metadata uses explicit `sample_id`, `image_path` and `target_name` pairs. The submission writer checks complete, unique pairs and matches `sample_submission.csv` IDs to test metadata while preserving the sample file's row order. It does not split IDs or infer target order from a matrix.

Python records images in first-occurrence metadata order. `features.csv` starts with `image_path`, followed by `feature_000001` and subsequent numeric columns. `features.npz` contains `features` and `image_path`; `manifest.csv` records one-based row indices and image paths. `schema.json` records model identity, processor settings, actual feature width, grid, crop, tile order and ordered blocks with zero-based, end-exclusive offsets. R matches observations, features, folds and predictions by `image_path`. An unidentified matrix needs its known original manifest before it can be used; row order is not reconstructed from a pivot or filename sort.

`config/report.R` separates raw `input_root`, `feature_root` and `output_dir`. `CSIRO_DATA_ROOT` changes the raw-input root; its default is the checkout containing `data/`. Feature paths remain relative to `feature_root`, defaulting to `features/`. Explicit absolute read paths are also accepted. Outputs and caches are confined to the checkout; set R's startup temporary-directory variables as shown in the README.

**Frozen feature extraction**

DINOv2 uses local Hugging Face processor and model files, `local_files_only=True`, frozen parameters and evaluation mode. The supported checkpoint type is ordinary DINOv2 with one CLS prefix. Resizing and center cropping are disabled in the processor. Its other local processor settings are retained and recorded, including rescaling and normalization; no additional pixel or embedding normalization is inserted. Whole-image feature order is **mean patch-token embedding, then `pooler_output`**.

Tiled extraction retains the report's 1000 × 2000 image assumption and patch size 14. A central crop makes dimensions divisible by both the supplied row/column grid and patch size. Tiles are traversed row-major within each original image. All patch tokens from all its tiles form one global token collection; tile pooler outputs are averaged separately. The basic tiled-mean implementation concatenates the global patch mean and mean tile pooler output. Grid dimensions must be given explicitly; tile counts in historical tables do not establish the grid layout.

The advanced tiled descriptor concatenates 17 embedding-width blocks:

| Order | Blocks |
| --- | --- |
| 1–7 | Mean, sample standard deviation, skewness, non-excess kurtosis, GeM, minimum, maximum |
| 8–12 | Quantiles at 0.10, 0.25, 0.50, 0.75, 0.90 |
| 13–17 | Positive proportion, L2 norm, L1 norm, L1/L2 ratio, mean tile pooler output |

Standard deviation uses the sample convention. Skewness and kurtosis use centered third/fourth moments divided by `std^3 + 1e-6` and `std^4 + 1e-6`. GeM clamps tokens below at `1e-6` and uses p = 3. Norms aggregate over the global token dimension; the L1/L2 denominator adds `1e-6`. Widths come from actual tensors, not a fixed dimension or a count of independently pooled tiles.

ConvNeXt uses Python/Keras, TensorFlow and matching local backbone weights. Architecture (`large`/`xlarge`), built-in preprocessing and pixel scaling are explicit choices. Global-average output width is read from the selected model. `--stage-layers STAGE2 STAGE3` requests mean/max pooling from two existing layer names; names must match that backbone. `--zoom` enables the supplied central-crop variant and TensorFlow bilinear resizing. Initial image loading retains Keras nearest-neighbour resizing. The implementation follows the [Keras ConvNeXt API](https://keras.io/api/applications/convnext/convnext_models/); raw 0–255 and `divide_255` input profiles remain separately selectable. Weight compatibility and numerical agreement are not established.

**PCA, regression and validation**

Main evaluation keeps all images sharing `Sampling_Date` together in four folds. It accepts `image_path,fold` assignments through `fold_manifest`; with `NULL`, it assembles a new seeded date-group split and saves it. Seed 123 does not recover the missing historical assignments. Grouping is neither forward-in-time validation nor a geographical holdout guarantee; no state-per-date assumption is imposed.

Each training partition removes zero-variance feature columns and fits centered, scaled PCA. Exactly 50 components are retained, with an explicit error if the partition has insufficient columns, samples or numerical rank. Validation rows use that fitted transformation. `e1071::svm` uses epsilon regression, radial kernels and `scale = TRUE`, without log-target transforms. The source report parameters are:

| Target | log10(cost) | log10(gamma) | log10(epsilon) |
| --- | ---: | ---: | ---: |
| Green | 2.129183135 | −3.23883311 | −2.999913551 |
| Dead | 0.8902861 | −1.8498431 | −2.796425884 |
| Clover | 3.0055243 | −3.6351474 | −2.415722 |

The separate historical tuning profile fits PCA once before resampling. It searches log10 cost in [−2, 4], gamma in [−5, 1] and epsilon in [−3, 0], with 80 MBO iterations and expected improvement. It uses the [mlr MBO control interface](https://mlr.mlr-org.com/reference/makeTuneControlMBO.html) and monitors component R² and support-vector counts. It does not provide fold-local preprocessing during tuning and is not the default evaluation path. Historical leaderboard scores do not describe results from new splits.

**Separate experiments**

`experiments/` contains a metadata baseline, explicit-channel colour summaries and mask, a column-selection/PCA helper, boosting variants and neural-head constructors. State, species, GreenSeeker NDVI and pasture height are training metadata, not ordinary test predictors for the image-only submission. `(G - R) / (G + R + 1e-6)` is an RGB colour proxy, not measured NDVI. Colour helpers require explicit red/green/blue planes; channel order must be known. The optional block selector is not activated by the main pipeline. Boosting's prepared PCA and early-stopping validation are separate experimental procedures; neural constructors are not the submitted frozen-feature method.

No checkpoint, feature array, historical fold manifest, fitted model, submission receipt or interactive UMAP is supplied.

The source reports cite Liao et al., *Estimating Pasture Biomass from Top-View Images* (2025); Liu et al., *A ConvNet for the 2020s* (2022); Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023); Smola and Schölkopf, *A Tutorial on Support Vector Regression* (2004); and Bischl et al., *mlrMBO* (2017). Model and data rights remain with their respective authors and providers.
