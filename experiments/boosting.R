# lm_xg.R: an LM plus XGBoost fitted to residuals on already prepared predictors.
# Validation labels guide early stopping, so this is an exploratory fit rather than
# an independent assessment of a model chosen without those labels.
fit_linear_residual_boosting <- function(
    predictors, target, training_rows,
    parameters = list(
      booster = "gbtree", eta = 0.00025013058, max_depth = 9,
      subsample = 0.6, colsample_bytree = 0.5, min_child_weight = 2.07191,
      gamma = 0.000835919, objective = "reg:squarederror", eval_metric = "rmse"
    ), nrounds = 15000, early_stopping_rounds = 200) {
  training_data <- data.frame(y = target[training_rows], predictors[training_rows, , drop = FALSE])
  validation_data <- data.frame(y = target[-training_rows], predictors[-training_rows, , drop = FALSE])
  linear_model <- stats::lm(y ~ ., data = training_data)
  training_predictions <- stats::predict(linear_model, training_data)
  validation_predictions <- stats::predict(linear_model, validation_data)
  training_matrix <- xgboost::xgb.DMatrix(
    data = as.matrix(predictors[training_rows, , drop = FALSE]),
    label = target[training_rows] - training_predictions
  )
  validation_matrix <- xgboost::xgb.DMatrix(
    data = as.matrix(predictors[-training_rows, , drop = FALSE]),
    label = target[-training_rows] - validation_predictions
  )
  boosting_model <- xgboost::xgb.train(
    params = parameters, data = training_matrix, nrounds = nrounds,
    early_stopping_rounds = early_stopping_rounds,
    watchlist = list(dtest = validation_matrix)
  )
  residual_predictions <- stats::predict(boosting_model, as.matrix(predictors[-training_rows, , drop = FALSE]))
  list(
    linear_model = linear_model, boosting_model = boosting_model,
    linear_predictions = validation_predictions,
    residual_predictions = residual_predictions,
    predictions = validation_predictions + residual_predictions
  )
}

# xgb_new_param_correct-split.R used scaled PCA before four supplied splits.
# feature_matrix must already be aligned to train_wide; missing historical indices
# are explicit inputs here rather than an assumed tuned object in an R session.
evaluate_global_pca_boosting <- function(feature_matrix, train_wide, training_indices) {
  if (length(training_indices) != 4L) stop("This source variant uses four splits.")
  pca <- stats::prcomp(feature_matrix, scale. = TRUE)
  predictors <- pca$x[, 1:50, drop = FALSE]
  parameters <- list(
    booster = "gbtree", eta = 0.001, max_depth = 50, subsample = 1,
    colsample_bytree = 1, min_child_weight = 2, gamma = 0,
    objective = "reg:squarederror", eval_metric = "rmse"
  )
  component_columns <- c("target.Dry_Clover_g", "target.Dry_Green_g", "target.Dry_Dead_g")
  target_columns <- c(
    "target.Dry_Green_g", "target.Dry_Dead_g", "target.Dry_Clover_g",
    "target.GDM_g", "target.Dry_Total_g"
  )
  fold_results <- lapply(training_indices, function(training_rows) {
    component_predictions <- lapply(component_columns, function(target_column) {
      training_matrix <- xgboost::xgb.DMatrix(
        data = as.matrix(predictors[training_rows, , drop = FALSE]),
        label = train_wide[[target_column]][training_rows]
      )
      validation_matrix <- xgboost::xgb.DMatrix(
        data = as.matrix(predictors[-training_rows, , drop = FALSE]),
        label = train_wide[[target_column]][-training_rows]
      )
      model <- xgboost::xgb.train(
        params = parameters, data = training_matrix, nrounds = 15000,
        early_stopping_rounds = 200, watchlist = list(dtest = validation_matrix)
      )
      stats::predict(model, predictors[-training_rows, , drop = FALSE])
    })
    names(component_predictions) <- component_columns
    predictions <- as.data.frame(component_predictions)
    predictions[predictions < 0] <- 0
    predictions$target.GDM_g <- predictions$target.Dry_Green_g + predictions$target.Dry_Clover_g
    predictions$target.Dry_Total_g <- predictions$target.Dry_Green_g +
      predictions$target.Dry_Dead_g + predictions$target.Dry_Clover_g
    predictions <- predictions[, target_columns, drop = FALSE]
    actual <- as.matrix(train_wide[-training_rows, target_columns, drop = FALSE])
    weights <- matrix(c(0.1, 0.1, 0.1, 0.2, 0.5), nrow(actual), 5, byrow = TRUE)
    global_mean <- 5 * mean(actual * weights)
    weighted_r2 <- 1 - sum((actual - as.matrix(predictions))^2 * weights) /
      sum((actual - global_mean)^2 * weights)
    list(
      image_path = train_wide$image_path[-training_rows],
      predictions = predictions, weighted_r2 = weighted_r2
    )
  })
  list(pca = pca, folds = fold_results)
}

# automl_boost.R tunes the green target on supplied PCA coordinates, with date groups.
tune_green_boosting <- function(pca_features, train_wide) {
  task <- mlr::makeRegrTask(
    data = data.frame(pca_features, target = train_wide$target.Dry_Green_g),
    target = "target", blocking = as.factor(train_wide$Sampling_Date)
  )
  learner <- mlr::makeLearner(
    "regr.xgboost", predict.type = "response",
    par.vals = list(objective = "reg:squarederror")
  )
  parameter_set <- ParamHelpers::makeParamSet(
    ParamHelpers::makeNumericParam("nrounds", lower = 2, upper = 3.4,
      trafo = function(value) as.integer(round(10^value))),
    ParamHelpers::makeNumericParam("eta", lower = -4, upper = 0,
      trafo = function(value) 10^value),
    ParamHelpers::makeIntegerParam("max_depth", lower = 2, upper = 10),
    ParamHelpers::makeNumericParam("subsample", lower = 0.5, upper = 1),
    ParamHelpers::makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
    ParamHelpers::makeNumericParam("min_child_weight", lower = 1, upper = 10),
    ParamHelpers::makeNumericParam("gamma", lower = 0, upper = 5)
  )
  resampling <- mlr::makeResampleDesc("CV", iters = 4, blocking.cv = TRUE)
  control <- mlrMBO::setMBOControlTermination(
    mlrMBO::setMBOControlInfill(
      mlrMBO::makeMBOControl(), crit = mlrMBO::makeMBOInfillCritEI()
    ), iters = 80
  )
  set.seed(123)
  tuned <- mlr::tuneParams(
    learner = learner, task = task, resampling = resampling,
    measures = list(mlr::rsq), par.set = parameter_set,
    control = mlr::makeTuneControlMBO(mbo.control = control), show.info = TRUE
  )
  list(tuned = tuned, optimization_path = as.data.frame(tuned$opt.path))
}
