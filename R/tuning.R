# Historical report profile: PCA is fitted once before grouped resampling.
# This profile is separate from the fold-local PCA used by evaluate_biomass_cv.
tune_svr_global_pca <- function(features, metadata, manifest, target_name,
                                 components = 50L, seed = 123L) {
  packages <- c("mlr", "mlrMBO", "ParamHelpers", "DiceKriging", "e1071")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Tuning requires: ", paste(missing, collapse = ", "))
  if (!target_name %in% component_targets()) stop("Tune one of the three component targets.")
  manifest <- validate_folds(metadata, manifest)
  preprocessing <- fit_feature_pca(features, components)
  transformed <- transform_feature_pca(preprocessing, features)
  task <- mlr::makeRegrTask(data = data.frame(transformed, target = metadata[[target_name]]),
                            target = "target", blocking = as.factor(metadata$Sampling_Date))
  learner <- mlr::makeLearner("regr.svm", type = "eps-regression", kernel = "radial", scale = TRUE)
  parameter_set <- ParamHelpers::makeParamSet(
    ParamHelpers::makeNumericParam("cost", lower = -2, upper = 4, trafo = function(x) 10^x),
    ParamHelpers::makeNumericParam("gamma", lower = -5, upper = 1, trafo = function(x) 10^x),
    ParamHelpers::makeNumericParam("epsilon", lower = -3, upper = 0, trafo = function(x) 10^x)
  )
  description <- mlr::makeResampleDesc("CV", iters = 4, blocking.cv = TRUE)
  set.seed(seed)
  resampling <- mlr::makeResampleInstance(description, task = task)
  resampling$train.inds <- lapply(seq_len(4), function(fold) which(manifest$fold != fold))
  resampling$test.inds <- lapply(seq_len(4), function(fold) which(manifest$fold == fold))
  support_vector_measure <- mlr::makeMeasure(
    id = "nSV", name = "Number of Support Vectors", minimize = TRUE, best = 0, worst = Inf,
    properties = c("classif", "regr", "req.model"),
    fun = function(task, model, pred, feats, extra.args) {
      fitted <- mlr::getLearnerModel(model, more.unwrap = TRUE)
      fitted$tot.nSV
    }
  )
  control <- mlr::makeTuneControlMBO(mbo.control = mlrMBO::setMBOControlTermination(
    mlrMBO::setMBOControlInfill(mlrMBO::makeMBOControl(), crit = mlrMBO::makeMBOInfillCritEI()),
    iters = 80
  ))
  tuned <- mlr::tuneParams(learner = learner, task = task, resampling = resampling,
                           measures = list(mlr::rsq, support_vector_measure), par.set = parameter_set,
                           control = control, show.info = TRUE)
  list(tuned = tuned, preprocessing = preprocessing, target_name = target_name,
       folds = manifest, profile = "historical_global_pca")
}
