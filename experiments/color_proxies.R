# Supply channel planes explicitly; the magick array layout must be known.
color_distribution <- function(values, include_extremes = TRUE) {
  summaries <- c(
    mean = mean(values, na.rm = TRUE),
    sd = stats::sd(values, na.rm = TRUE)
  )
  if (include_extremes) {
    summaries <- c(summaries, min = min(values, na.rm = TRUE), max = max(values, na.rm = TRUE))
  }
  c(
    summaries,
    median = stats::median(values, na.rm = TRUE),
    q10 = unname(stats::quantile(values, 0.1, na.rm = TRUE)),
    q25 = unname(stats::quantile(values, 0.25, na.rm = TRUE)),
    q75 = unname(stats::quantile(values, 0.75, na.rm = TRUE)),
    q90 = unname(stats::quantile(values, 0.9, na.rm = TRUE))
  )
}

# Channel values are the source's 0-1 intensities. This RGB proxy is not GreenSeeker NDVI.
# old and expanded retain the two colour-statistic sets; proxy retains ndvi.R's set.
color_proxy_features <- function(red, green, blue, variant = c("expanded", "old", "proxy")) {
  variant <- match.arg(variant)
  green_red_proxy <- (green - red) / (green + red + 1e-6)
  proxy_statistics <- color_distribution(green_red_proxy)
  names(proxy_statistics) <- paste0("green_red_", names(proxy_statistics))
  if (variant == "proxy") {
    return(proxy_statistics[c(
      "green_red_mean", "green_red_sd", "green_red_min", "green_red_max",
      "green_red_q10", "green_red_q90", "green_red_median",
      "green_red_q25", "green_red_q75"
    )])
  }

  channel_statistics <- unlist(lapply(
    list(red = red, green = green, blue = blue),
    color_distribution, include_extremes = variant == "old"
  ))
  names(channel_statistics) <- gsub(".", "_", names(channel_statistics), fixed = TRUE)
  if (variant == "old") {
    return(c(proxy_statistics, channel_statistics))
  }

  vari <- (green - red) / (green + red - blue + 1e-6)
  excess_green <- 2 * green - red - blue
  thresholds <- c(0, -0.058, -0.025, 0.02, 0.09, 0.16, 0.22)
  proportions <- vapply(thresholds, function(threshold) {
    # The green-red threshold proportions did not remove NA in color_features.R.
    mean(green_red_proxy > threshold)
  }, numeric(1))
  names(proportions) <- c(
    "green_red_above_0", "green_red_above_minus_0_058", "green_red_above_minus_0_025",
    "green_red_above_0_02", "green_red_above_0_09", "green_red_above_0_16",
    "green_red_above_0_22"
  )
  vari_statistics <- c(above_0 = mean(vari > 0, na.rm = TRUE), color_distribution(vari))
  names(vari_statistics) <- paste0("vari_", names(vari_statistics))
  excess_statistics <- c(
    above_0 = mean(excess_green > 0, na.rm = TRUE), color_distribution(excess_green)
  )
  names(excess_statistics) <- paste0("excess_green_", names(excess_statistics))
  c(
    proportions[1], proxy_statistics, proportions[-1], vari_statistics,
    excess_statistics, channel_statistics
  )
}

# Plane-wise arithmetic from mask.R, without guessing its image-array channel axis.
green_dominance_mask <- function(red, green, blue) {
  mask <- ifelse(green > red & green > blue, 1, 0)
  list(mask = mask, red = red * mask, green = green * mask, blue = blue * mask)
}
