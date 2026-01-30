# Architecture only from nn.R: species classification and standardized NDVI/height
# regression. Training metadata and the source's unbounded prediction generator
# are not part of the image-only biomass submission workflow.
build_metadata_cnn <- function(number_of_species, image_shape = c(250L, 500L, 3L)) {
  inputs <- keras3::layer_input(shape = image_shape)
  encoded <- keras3::layer_conv_2d(inputs, 32, 3, padding = "same")
  encoded <- keras3::layer_batch_normalization(encoded)
  encoded <- keras3::layer_activation(encoded, "relu")
  encoded <- keras3::layer_conv_2d(encoded, 32, 3, padding = "same")
  encoded <- keras3::layer_activation(encoded, "relu")
  encoded <- keras3::layer_max_pooling_2d(encoded, pool_size = c(2, 2))
  encoded <- keras3::layer_conv_2d(encoded, 64, 3, padding = "same")
  encoded <- keras3::layer_batch_normalization(encoded)
  encoded <- keras3::layer_activation(encoded, "relu")
  encoded <- keras3::layer_max_pooling_2d(encoded, pool_size = c(2, 2))
  encoded <- keras3::layer_conv_2d(encoded, 128, 3, padding = "same")
  encoded <- keras3::layer_batch_normalization(encoded)
  encoded <- keras3::layer_activation(encoded, "relu")
  encoded <- keras3::layer_global_average_pooling_2d(encoded)
  encoded <- keras3::layer_dropout(encoded, 0.4)
  species_output <- keras3::layer_dense(
    encoded, units = number_of_species, activation = "softmax", name = "species"
  )
  features_output <- keras3::layer_dense(encoded, units = 2, activation = "linear", name = "features")
  keras3::keras_model(
    inputs = inputs, outputs = list(species = species_output, features = features_output)
  )
}

# nn2.R's frozen EfficientNet-B0 head. Supply an already loaded local backbone;
# this constructor neither downloads the original ImageNet weights nor fits a model.
# Four standardized output targets retain the source order, including direct GDM.
build_efficientnet_biomass_head <- function(efficientnet_backbone, image_shape = c(224L, 224L, 3L)) {
  keras3::freeze_weights(efficientnet_backbone)
  inputs <- keras3::layer_input(shape = image_shape)
  augmentation <- keras3::keras_model_sequential()
  augmentation <- keras3::layer_random_rotation(augmentation, 0.2)
  augmentation <- keras3::layer_random_zoom(augmentation, 0.2)
  augmentation <- keras3::layer_random_contrast(augmentation, 0.2)
  encoded <- augmentation(inputs)
  encoded <- efficientnet_backbone(encoded)
  encoded <- keras3::layer_global_average_pooling_2d(encoded)
  encoded <- keras3::layer_batch_normalization(encoded)
  encoded <- keras3::layer_dropout(encoded, 0.5)
  encoded <- keras3::layer_dense(encoded, 128, activation = "relu")
  outputs <- keras3::layer_dense(encoded, 4, activation = "linear")
  list(
    model = keras3::keras_model(inputs, outputs),
    target_columns = c(
      "target.Dry_Clover_g", "target.Dry_Dead_g", "target.Dry_Green_g", "target.GDM_g"
    )
  )
}
