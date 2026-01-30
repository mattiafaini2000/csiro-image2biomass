# Train-only metadata baseline from csiro.R. These predictors are not in test.csv.
metadata_species_split <- function(train_wide, proportion = 0.8) {
  caret::createDataPartition(train_wide$Species, p = proportion)$Resample1
}

# Inputs use the source's target.<name> columns; training_rows is an explicit split.
# Predictions retain the source's absence of non-negativity clipping.
fit_metadata_baseline <- function(train_wide, training_rows) {
  component_model <- stats::lm(
    cbind(target.Dry_Clover_g, target.Dry_Dead_g, target.Dry_Green_g) ~
      Pre_GSHH_NDVI + Height_Ave_cm + Species,
    data = train_wide[training_rows, , drop = FALSE]
  )
  validation_data <- train_wide[-training_rows, , drop = FALSE]
  predictions <- data.frame(stats::predict(component_model, newdata = validation_data))
  predictions$target.Dry_Total_g <- predictions$target.Dry_Clover_g +
    predictions$target.Dry_Dead_g + predictions$target.Dry_Green_g
  predictions$target.GDM_g <- predictions$target.Dry_Clover_g +
    predictions$target.Dry_Green_g
  target_columns <- c(
    "target.Dry_Green_g", "target.Dry_Dead_g", "target.Dry_Clover_g",
    "target.GDM_g", "target.Dry_Total_g"
  )
  actual <- as.matrix(validation_data[, target_columns, drop = FALSE])
  predictions <- predictions[, target_columns, drop = FALSE]
  weights <- matrix(c(0.1, 0.1, 0.1, 0.2, 0.5), nrow(actual), 5, byrow = TRUE)
  global_mean <- 5 * mean(actual * weights)
  weighted_r2 <- 1 - sum((actual - as.matrix(predictions))^2 * weights) /
    sum((actual - global_mean)^2 * weights)
  list(
    model = component_model,
    image_path = validation_data$image_path,
    predictions = predictions,
    weighted_r2 = weighted_r2
  )
}
