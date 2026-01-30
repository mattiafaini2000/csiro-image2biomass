arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2L) {
  stop("Usage: Rscript --vanilla scripts/tune_svr.R config/report.R Dry_Dead_g")
}
script_file <- sub("^--file=", "", commandArgs()[grepl("^--file=", commandArgs())][1])
project_root <- normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = TRUE)
for (module in c("data", "features", "preprocessing", "resampling", "svr", "tuning")) {
  source(file.path(project_root, "R", paste0(module, ".R")))
}
config <- load_analysis_config(arguments[1], project_root)
metadata <- read_training_metadata(input_path(config, config$train_metadata))
features <- align_features(read_features(input_path(config, config$train_features, config$feature_root)), metadata$image_path)
manifest <- load_folds(config, metadata)
tuning <- tune_svr_global_pca(features, metadata, manifest, arguments[2], config$components, config$seed)
output_dir <- output_directory(config, file.path("tuning_global_pca", arguments[2]))
saveRDS(tuning, file.path(output_dir, "tuning.rds"))
utils::write.csv(as.data.frame(tuning$tuned$opt.path), file.path(output_dir, "optimization_path.csv"), row.names = FALSE)
utils::write.csv(manifest, file.path(output_dir, "folds.csv"), row.names = FALSE)
