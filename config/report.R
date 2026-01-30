# Optional external read-only root; outputs always stay in this repository.
analysis_config <- list(
  input_root = Sys.getenv("CSIRO_DATA_ROOT", unset = "."),
  train_metadata = "data/train.csv",
  test_metadata = "data/test.csv",
  sample_submission = "data/sample_submission.csv",
  feature_root = "features",
  train_features = "train/features.csv",
  test_features = "test/features.csv",
  fold_manifest = NULL,
  folds = 4L,
  seed = 123L,
  components = 50L,
  output_dir = "outputs",
  parameter_source = "report/report.Rmd: final-cv-evaluation",
  parameters = list(
    Dry_Green_g = list(cost = 10^2.129183135, gamma = 10^-3.23883311, epsilon = 10^-2.999913551),
    Dry_Dead_g = list(cost = 10^0.8902861, gamma = 10^-1.8498431, epsilon = 10^-2.796425884),
    Dry_Clover_g = list(cost = 10^3.0055243, gamma = 10^-3.6351474, epsilon = 10^-2.415722e+00)
  )
)
