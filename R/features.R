# Python interchange: image_path followed by finite numeric feature columns.
read_features <- function(file) {
  feature_table <- utils::read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
  if (ncol(feature_table) < 2L || names(feature_table)[1] != "image_path") {
    stop("Feature CSV must start with image_path and contain feature columns.")
  }
  if (anyDuplicated(names(feature_table)) || anyDuplicated(feature_table$image_path) ||
      anyNA(feature_table$image_path) || any(!nzchar(feature_table$image_path))) {
    stop("Feature identifiers and column names must be unique and non-missing.")
  }
  feature_columns <- feature_table[-1]
  if (!all(vapply(feature_columns, is.numeric, logical(1)))) {
    stop("Feature columns must be numeric.")
  }
  feature_matrix <- as.matrix(feature_columns)
  if (any(!is.finite(feature_matrix))) stop("Feature values must be finite.")
  list(image_path = feature_table$image_path, values = feature_matrix)
}

align_features <- function(features, image_paths) {
  positions <- match(image_paths, features$image_path)
  if (anyNA(positions)) stop("Features are missing for requested image_path values.")
  features$values[positions, , drop = FALSE]
}
