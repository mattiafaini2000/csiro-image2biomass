biomass_weights <- function() {
  c(Dry_Green_g = 0.1, Dry_Dead_g = 0.1, Dry_Clover_g = 0.1, GDM_g = 0.2, Dry_Total_g = 0.5)
}

# A single global weighted mean is used, not five separate target-wise R-squareds.
weighted_global_r_squared <- function(truth, predictions) {
  weights <- biomass_weights()
  require_columns(truth, names(weights), "Observed targets")
  require_columns(predictions, names(weights), "Predicted targets")
  truth <- as.matrix(truth[, names(weights), drop = FALSE])
  predictions <- as.matrix(predictions[, names(weights), drop = FALSE])
  if (!identical(dim(truth), dim(predictions)) || !nrow(truth) ||
      any(!is.finite(truth)) || any(!is.finite(predictions))) {
    stop("Scoring requires equally sized finite target matrices.")
  }
  global_mean <- sum(sweep(truth, 2, weights, "*")) / (nrow(truth) * sum(weights))
  residual_sum <- sum(sweep((truth - predictions)^2, 2, weights, "*"))
  total_sum <- sum(sweep((truth - global_mean)^2, 2, weights, "*"))
  if (total_sum == 0) stop("Weighted R-squared is undefined for zero total variation.")
  1 - residual_sum / total_sum
}
