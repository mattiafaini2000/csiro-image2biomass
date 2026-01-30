# Match test.csv's explicit sample IDs and target names, retaining sample-submission order.
make_submission <- function(predictions, test_metadata, sample_submission) {
  validate_test_metadata(test_metadata)
  require_columns(predictions, c("image_path", biomass_targets()), "Image predictions")
  require_columns(sample_submission, c("sample_id", "target"), "sample_submission.csv")
  if (anyDuplicated(predictions$image_path) || anyDuplicated(sample_submission$sample_id) ||
      anyNA(sample_submission$sample_id) ||
      any(!nzchar(trimws(as.character(sample_submission$sample_id)))) ||
      !setequal(sample_submission$sample_id, test_metadata$sample_id)) {
    stop("Submission IDs must match test.csv exactly and image predictions must be unique.")
  }
  ordered_test <- test_metadata[match(sample_submission$sample_id, test_metadata$sample_id), ]
  image_rows <- match(ordered_test$image_path, predictions$image_path)
  if (anyNA(image_rows)) stop("Predictions are missing for test images.")
  target_columns <- match(ordered_test$target_name, biomass_targets())
  prediction_matrix <- as.matrix(predictions[, biomass_targets(), drop = FALSE])
  values <- prediction_matrix[cbind(image_rows, target_columns)]
  if (any(!is.finite(values)) || any(values < 0)) stop("Submission targets must be finite and non-negative.")
  data.frame(sample_id = sample_submission$sample_id, target = values, stringsAsFactors = FALSE)
}
