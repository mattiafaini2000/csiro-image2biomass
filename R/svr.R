component_targets <- function() c("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g")

fit_component_svrs <- function(features, metadata, parameters) {
  if (!requireNamespace("e1071", quietly = TRUE)) stop("The e1071 package is required.")
  targets <- component_targets()
  if (!setequal(names(parameters), targets)) stop("Provide parameters for all three component targets.")
  models <- lapply(targets, function(target) {
    settings <- parameters[[target]]
    do.call(e1071::svm, c(list(x = features, y = metadata[[target]],
                             type = "eps-regression", kernel = "radial", scale = TRUE), settings))
  })
  stats::setNames(models, targets)
}

predict_components <- function(models, features) {
  as.data.frame(lapply(models, function(model) as.numeric(stats::predict(model, features))),
                check.names = FALSE)
}

# Clip the components first; then derive GDM and total biomass, all in grams.
derive_biomass_targets <- function(components) {
  require_columns(components, component_targets(), "Component predictions")
  predictions <- components[, component_targets(), drop = FALSE]
  predictions[predictions < 0] <- 0
  predictions$GDM_g <- predictions$Dry_Green_g + predictions$Dry_Clover_g
  predictions$Dry_Total_g <- predictions$Dry_Green_g + predictions$Dry_Dead_g + predictions$Dry_Clover_g
  predictions[, biomass_targets(), drop = FALSE]
}

fit_biomass_model <- function(features, metadata, parameters, components = 50L) {
  preprocessing <- fit_feature_pca(features, components)
  transformed <- transform_feature_pca(preprocessing, features)
  list(preprocessing = preprocessing,
       models = fit_component_svrs(transformed, metadata, parameters), parameters = parameters)
}

predict_biomass_model <- function(model, features) {
  transformed <- transform_feature_pca(model$preprocessing, features)
  derive_biomass_targets(predict_components(model$models, transformed))
}

evaluate_biomass_cv <- function(features, metadata, manifest, parameters, components = 50L) {
  manifest <- validate_folds(metadata, manifest)
  predictions <- as.data.frame(matrix(NA_real_, nrow(metadata), length(biomass_targets())))
  names(predictions) <- biomass_targets()
  fold_scores <- numeric(4)
  support_vectors <- list()
  for (fold in seq_len(4L)) {
    validation_rows <- which(manifest$fold == fold)
    training_rows <- which(manifest$fold != fold)
    model <- fit_biomass_model(features[training_rows, , drop = FALSE],
                               metadata[training_rows, , drop = FALSE], parameters, components)
    fold_predictions <- predict_biomass_model(model, features[validation_rows, , drop = FALSE])
    predictions[validation_rows, ] <- fold_predictions
    fold_scores[fold] <- weighted_global_r_squared(metadata[validation_rows, ], fold_predictions)
    support_vectors[[fold]] <- data.frame(
      fold = fold, target_name = names(model$models),
      support_vectors = vapply(model$models, function(component) component$tot.nSV, numeric(1))
    )
  }
  list(fold_scores = data.frame(fold = seq_len(4L), weighted_r_squared = fold_scores),
       summary = data.frame(statistic = c("mean_four_fold_scores", "pooled_out_of_fold_score"),
                            weighted_r_squared = c(mean(fold_scores),
                              weighted_global_r_squared(metadata, predictions))),
       predictions = data.frame(image_path = metadata$image_path, predictions, check.names = FALSE),
       support_vectors = do.call(rbind, support_vectors))
}
