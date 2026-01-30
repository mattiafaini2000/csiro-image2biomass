# Fit on the training partition only. Preserve exactly 50 PCs or fail explicitly.
fit_feature_pca <- function(features, components = 50L) {
  retained <- vapply(seq_len(ncol(features)), function(column) {
    stats::var(features[, column]) > 0
  }, logical(1))
  if (anyNA(retained) || sum(retained) < components || nrow(features) - 1L < components) {
    stop("Training features cannot support the required ", components, " principal components.")
  }
  pca <- stats::prcomp(features[, retained, drop = FALSE], center = TRUE, scale. = TRUE)
  numerical_rank <- sum(pca$sdev > sqrt(.Machine$double.eps) * pca$sdev[1])
  if (numerical_rank < components) {
    stop("Training feature rank is below the required ", components, " principal components.")
  }
  list(pca = pca, columns = colnames(features)[retained], components = components)
}

transform_feature_pca <- function(preprocessing, features) {
  if (!all(preprocessing$columns %in% colnames(features))) {
    stop("Features do not match the fitted PCA column schema.")
  }
  stats::predict(preprocessing$pca, features[, preprocessing$columns, drop = FALSE])[
    , seq_len(preprocessing$components), drop = FALSE
  ]
}
