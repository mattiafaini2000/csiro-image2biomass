# Date-grouped split; historical fold assignments are not supplied.
make_date_folds <- function(metadata, folds = 4L, seed = 123L) {
  dates <- sort(unique(metadata$Sampling_Date))
  if (length(dates) < folds) stop("There must be at least one distinct date per fold.")
  set.seed(seed)
  shuffled_dates <- sample(dates, length(dates))
  date_folds <- rep(seq_len(folds), length.out = length(shuffled_dates))
  data.frame(image_path = metadata$image_path,
             fold = date_folds[match(metadata$Sampling_Date, shuffled_dates)],
             stringsAsFactors = FALSE)
}

validate_folds <- function(metadata, manifest, folds = 4L) {
  require_columns(manifest, c("image_path", "fold"), "Fold manifest")
  if (anyDuplicated(manifest$image_path) ||
      !setequal(manifest$image_path, metadata$image_path)) {
    stop("Fold manifest must contain each training image exactly once.")
  }
  manifest <- manifest[match(metadata$image_path, manifest$image_path), c("image_path", "fold")]
  if (anyNA(manifest$fold) || any(manifest$fold != as.integer(manifest$fold)) ||
      !setequal(unique(manifest$fold), seq_len(folds))) {
    stop("Fold manifest must use each of the requested integer fold labels.")
  }
  grouped <- split(manifest$fold, metadata$Sampling_Date)
  if (any(vapply(grouped, function(values) length(unique(values)) != 1L, logical(1)))) {
    stop("Images sharing Sampling_Date must belong to the same fold.")
  }
  manifest
}

load_folds <- function(config, metadata) {
  if (is.null(config$fold_manifest)) {
    manifest <- make_date_folds(metadata, config$folds, config$seed)
  } else {
    manifest <- utils::read.csv(input_path(config, config$fold_manifest), stringsAsFactors = FALSE)
  }
  validate_folds(metadata, manifest, config$folds)
}
