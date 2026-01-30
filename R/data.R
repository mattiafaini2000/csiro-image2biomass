biomass_targets <- function() {
  c("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")
}

require_columns <- function(data, columns, description) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) {
    stop(description, " is missing: ", paste(missing, collapse = ", "))
  }
  invisible(data)
}

# Return one row per image; target values are in grams.
read_training_metadata <- function(file) {
  observations <- utils::read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
  require_columns(observations, c("image_path", "target_name", "target", "Sampling_Date"), "train.csv")
  targets <- biomass_targets()
  if (anyNA(observations[c("image_path", "target_name", "Sampling_Date")]) ||
      any(!nzchar(observations$image_path)) ||
      any(!nzchar(as.character(observations$Sampling_Date))) ||
      any(!observations$target_name %in% targets)) {
    stop("Training identifiers, dates and target names must be present and valid.")
  }
  if (!is.numeric(observations$target) || any(!is.finite(observations$target))) {
    stop("Training targets must be finite numeric values.")
  }
  pairs <- observations[c("image_path", "target_name")]
  if (anyDuplicated(pairs)) stop("Duplicate image_path/target_name pair in training metadata.")
  image_paths <- unique(observations$image_path)
  dates <- vapply(image_paths, function(image_path) {
    values <- unique(observations$Sampling_Date[observations$image_path == image_path])
    if (length(values) != 1L) stop("Sampling_Date must be unique for each image.")
    as.character(values)
  }, character(1))
  metadata <- data.frame(image_path = image_paths, Sampling_Date = unname(dates),
                         stringsAsFactors = FALSE)
  for (target_name in targets) {
    target_rows <- observations[observations$target_name == target_name, ]
    positions <- match(image_paths, target_rows$image_path)
    if (anyNA(positions)) stop("Every training image needs all five biomass targets.")
    metadata[[target_name]] <- target_rows$target[positions]
  }
  metadata
}

read_test_metadata <- function(file) {
  metadata <- utils::read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
  validate_test_metadata(metadata)
  metadata
}

validate_test_metadata <- function(metadata) {
  require_columns(metadata, c("sample_id", "image_path", "target_name"), "test.csv")
  if (anyNA(metadata[c("sample_id", "image_path", "target_name")]) ||
      any(!nzchar(trimws(as.character(metadata$sample_id)))) ||
      any(!nzchar(trimws(as.character(metadata$image_path)))) ||
      anyDuplicated(metadata$sample_id) ||
      anyDuplicated(metadata[c("image_path", "target_name")]) ||
      any(!metadata$target_name %in% biomass_targets())) {
    stop("test.csv needs unique sample IDs and image/target pairs with known target names.")
  }
  target_sets <- split(metadata$target_name, metadata$image_path)
  if (!length(target_sets) || any(!vapply(target_sets, function(targets) {
    setequal(targets, biomass_targets()) && length(targets) == length(biomass_targets())
  }, logical(1)))) {
    stop("Every test image must have exactly one pair for each of the five targets.")
  }
  invisible(metadata)
}

load_analysis_config <- function(file, project_root) {
  configuration <- new.env(parent = baseenv())
  sys.source(file, envir = configuration)
  if (!exists("analysis_config", envir = configuration, inherits = FALSE)) {
    stop("Configuration must define analysis_config.")
  }
  config <- configuration$analysis_config
  config$project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
  config
}

input_path <- function(config, relative_path, input_root = config$input_root) {
  if (grepl("^([A-Za-z]:[/\\\\]|/|\\\\\\\\)", relative_path)) {
    return(normalizePath(relative_path, winslash = "/", mustWork = TRUE))
  }
  data_root <- input_root
  if (is.null(data_root) || !nzchar(data_root)) data_root <- config$project_root
  if (!grepl("^([A-Za-z]:[/\\\\]|/|\\\\\\\\)", data_root)) {
    data_root <- file.path(config$project_root, data_root)
  }
  normalizePath(file.path(data_root, relative_path), winslash = "/", mustWork = TRUE)
}

# Reject external destinations before creating directories, including symlink parents.
output_directory <- function(config, workflow) {
  root <- normalizePath(config$project_root, winslash = "/", mustWork = TRUE)
  output_root <- config$output_dir
  if (is.null(output_root)) output_root <- "outputs"
  output_root <- gsub("\\\\", "/", output_root)
  if (any(strsplit(output_root, "/", fixed = TRUE)[[1]] == "..")) {
    stop("Output paths must not contain '..'.")
  }
  if (!grepl("^([A-Za-z]:/|/)", output_root)) output_root <- file.path(root, output_root)
  destination <- file.path(output_root, workflow)
  ancestor <- destination
  while (!file.exists(ancestor) && !dir.exists(ancestor)) ancestor <- dirname(ancestor)
  ancestor <- normalizePath(ancestor, winslash = "/", mustWork = TRUE)
  inside <- identical(tolower(ancestor), tolower(root)) ||
    startsWith(tolower(ancestor), paste0(tolower(root), "/"))
  if (!inside) stop("Outputs must remain inside the project directory.")
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  normalizePath(destination, winslash = "/", mustWork = TRUE)
}
