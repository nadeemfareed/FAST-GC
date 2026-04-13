library(lidR)
library(RANN)

# ============================================================
# USER SETTINGS
# ============================================================
input_dir <- "E:/DAP/Aucilla/ALS"   # your folder (recursive)

# Conversion behavior
CONVERT_LAZ_TO_LAS <- TRUE

# Output behavior
OVERWRITE_OUTPUT <- FALSE              # FALSE -> writes *_clean.las
SKIP_IF_OUTPUT_EXISTS <- TRUE
DELETE_ORIGINAL_AFTER_SUCCESS <- FALSE  # TRUE deletes original input after success (safe)

# --- Hard Z bounds (fast kill for extreme junk)
Z_MIN <- -5      # set 0 if you never expect below-ground points
Z_MAX <- 120     # set near expected max canopy + buffer for stronger cleaning

# --- SOR parameters (Statistical Outlier Removal)
K_SOR <- 20          # neighbors
Z_SCORE_THRESH <- 3  # 2.5–3.5 typical; lower = more aggressive

# --- Optional radius outlier removal (helps random speckles)
DO_RADIUS_FILTER <- TRUE
RADIUS <- 1.5        # meters
MIN_NB_IN_RADIUS <- 2  # keep points with >= this many neighbors inside radius

# ============================================================
# Helpers
# ============================================================

make_output_path <- function(infile, overwrite = FALSE) {
  out <- sub("\\.(laz|las)$", ".las", infile, ignore.case = TRUE)
  if (!overwrite) out <- sub("\\.las$", "_clean.las", out, ignore.case = TRUE)
  out
}

# ---- Statistical Outlier Removal (SOR) using kNN mean distance
sor_filter <- function(xyz, k = 20, z_thresh = 3) {
  # xyz: Nx3 matrix
  n <- nrow(xyz)
  if (n <= (k + 1)) return(rep(TRUE, n))  # not enough points to filter

  # k+1 because first neighbor is the point itself (distance 0)
  nn <- RANN::nn2(xyz, xyz, k = k + 1)

  # Drop self-distance (first column)
  d <- nn$nn.dists[, -1, drop = FALSE]

  # Mean distance to neighbors
  md <- rowMeans(d)

  # Robust-ish z-score using mean/sd (fast); works well for ALS speckle noise
  mu <- mean(md)
  sig <- sd(md)
  if (!is.finite(sig) || sig == 0) return(rep(TRUE, n))

  z <- (md - mu) / sig
  keep <- z <= z_thresh
  keep
}

# ---- Radius outlier removal (keep if enough neighbors within RADIUS)
radius_filter <- function(xyz, radius = 1.5, min_nb = 2) {
  n <- nrow(xyz)
  if (n < 5) return(rep(TRUE, n))

  # Use kNN as an approximation: find a moderate k and count how many are within radius
  k <- min(50, n - 1)
  nn <- RANN::nn2(xyz, xyz, k = k + 1)
  d <- nn$nn.dists[, -1, drop = FALSE]

  nb_in_r <- rowSums(d <= radius)
  keep <- nb_in_r >= min_nb
  keep
}

clean_als_noise <- function(las) {
  if (lidR::is.empty(las)) return(las)

  # 1) Hard Z bounds
  las <- lidR::filter_poi(las, Z >= Z_MIN & Z <= Z_MAX)
  if (lidR::is.empty(las)) return(las)

  # Extract XYZ matrix
  xyz <- as.matrix(las@data[, c("X", "Y", "Z")])
  n0 <- nrow(xyz)

  # 2) SOR keep mask
  keep_sor <- sor_filter(xyz, k = K_SOR, z_thresh = Z_SCORE_THRESH)

  # 3) Optional radius filter keep mask
  if (DO_RADIUS_FILTER) {
    keep_rad <- radius_filter(xyz, radius = RADIUS, min_nb = MIN_NB_IN_RADIUS)
    keep <- keep_sor & keep_rad
  } else {
    keep <- keep_sor
  }

  # Apply mask
  las <- lidR::filter_poi(las, keep)
  if (lidR::is.empty(las)) return(las)

  n1 <- lidR::npoints(las)
  cat("  Noise filter kept", n1, "of", n0, "points\n")

  las
}

# ============================================================
# Collect files (.las and .laz)
# ============================================================
files_all <- list.files(
  input_dir,
  pattern = "\\.(las|laz)$",
  recursive = TRUE,
  full.names = TRUE,
  ignore.case = TRUE
)

cat("Found", length(files_all), "LAS/LAZ files\n")

# ============================================================
# Main loop
# ============================================================
for (infile in files_all) {

  ext <- tolower(tools::file_ext(infile))
  cat("\n▶ Processing:", infile, "\n")

  # Skip LAZ if conversion disabled
  if (ext == "laz" && !CONVERT_LAZ_TO_LAS) {
    cat("  ⚠ LAZ found but CONVERT_LAZ_TO_LAS=FALSE. Skipping.\n")
    next
  }

  out_las <- make_output_path(infile, overwrite = OVERWRITE_OUTPUT)

  if (SKIP_IF_OUTPUT_EXISTS && file.exists(out_las)) {
    cat("  ⚠ Output exists, skipping:", out_las, "\n")
    next
  }

  # Read input (LAS or LAZ)
  las <- tryCatch(
    lidR::readLAS(infile),
    error = function(e) {
      cat("  ❌ Read failed:", conditionMessage(e), "\n")
      return(NULL)
    }
  )

  if (is.null(las) || lidR::is.empty(las)) {
    cat("  ❌ Empty/invalid file, skipping\n")
    next
  }

  n_before <- lidR::npoints(las)

  # Clean
  las_clean <- tryCatch(
    clean_als_noise(las),
    error = function(e) {
      cat("  ❌ Cleaning failed:", conditionMessage(e), "\n")
      return(NULL)
    }
  )

  if (is.null(las_clean) || lidR::is.empty(las_clean)) {
    cat("  ❌ Cleaned cloud is empty — keeping original\n")
    next
  }

  n_after <- lidR::npoints(las_clean)
  cat("  Points:", n_before, "→", n_after, " (removed", n_before - n_after, ")\n")

  # Write cleaned LAS
  ok_write <- tryCatch(
    {
      lidR::writeLAS(las_clean, out_las)
      TRUE
    },
    error = function(e) {
      cat("  ❌ Write failed:", conditionMessage(e), "\n")
      FALSE
    }
  )

  if (!ok_write || !file.exists(out_las)) {
    cat("  ⚠ Output not written; keeping original\n")
    next
  }

  cat("  ✅ Wrote cleaned LAS:", out_las, "\n")

  # Optional delete original input AFTER success
  if (DELETE_ORIGINAL_AFTER_SUCCESS) {
    if (OVERWRITE_OUTPUT) {
      cat("  ⚠ OVERWRITE_OUTPUT=TRUE, not deleting original (same path)\n")
    } else {
      file.remove(infile)
      cat("  🗑 Deleted original:", infile, "\n")
    }
  }

  rm(las, las_clean)
  gc()
}

cat("\n✔ Done: LAZ/LAS processing + SOR/radius noise removal finished.\n")
