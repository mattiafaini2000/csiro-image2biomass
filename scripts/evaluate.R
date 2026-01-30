arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) stop("Usage: Rscript --vanilla scripts/evaluate.R config/report.R")
script_file <- sub("^--file=", "", commandArgs()[grepl("^--file=", commandArgs())][1])
project_root <- normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = TRUE)
for (module in c("data", "features", "preprocessing", "resampling", "metrics", "svr")) {
  source(file.path(project_root, "R", paste0(module, ".R")))
}
config <- load_analysis_config(arguments[1], project_root)
metadata <- read_training_metadata(input_path(config, config$train_metadata))
features <- align_features(read_features(input_path(config, config$train_features, config$feature_root)), metadata$image_path)
manifest <- load_folds(config, metadata)
evaluation <- evaluate_biomass_cv(features, metadata, manifest, config$parameters, config$components)
output_dir <- output_directory(config, "evaluation")
utils::write.csv(manifest, file.path(output_dir, "folds.csv"), row.names = FALSE)
for (name in names(evaluation)) {
  utils::write.csv(evaluation[[name]], file.path(output_dir, paste0(name, ".csv")), row.names = FALSE)
}
