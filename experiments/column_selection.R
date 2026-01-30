# col_selection.R computes candidate block indices but never applies them.
# This function therefore removes zero-variance columns from the full supplied matrix.
exploratory_column_pca <- function(image_features) {
  candidate_indices <- c((1536 * 1 + 1):(1536 * 4), (1536 * 7 + 1):(1536 * 12))
  non_constant_columns <- apply(image_features, 2, stats::var) > 0
  cleaned_features <- image_features[, non_constant_columns, drop = FALSE]
  list(
    pca = stats::prcomp(cleaned_features, scale. = TRUE),
    retained_columns = which(non_constant_columns),
    unused_candidate_indices = candidate_indices
  )
}
