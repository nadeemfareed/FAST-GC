# FAST-GC example from R
# This script demonstrates how to call FAST-GC from R using system2().

cat("Checking FAST-GC availability...\n")
help_check <- system2("fastgc", args = "--help", stdout = TRUE, stderr = TRUE)
cat(paste(help_check, collapse = "\n"))
cat("\n\n")

# ------------------------------------------------------------------
# Example 1: minimal command
# ------------------------------------------------------------------

cmd <- "fastgc"

args_minimal <- c(
  "--in_path", "F:/lidar_data/USA",
  "--out_dir", "F:/FAST_GC_Test",
  "--sensor_mode", "ALS"
)

cat("Minimal FAST-GC command:\n")
cat(paste(c(cmd, args_minimal), collapse = " "))
cat("\n\n")

# Uncomment to run:
# system2(cmd, args = args_minimal)

# ------------------------------------------------------------------
# Example 2: fuller tiled workflow
# ------------------------------------------------------------------

args_full <- c(
  "--in_path", "F:/lidar_data/USA",
  "--out_dir", "F:/FAST_GC_Test",
  "--sensor_mode", "ALS",
  "--workflow", "tile-run-merge",
  "--products", "FAST_GC", "FAST_DEM", "FAST_NORMALIZED", "FAST_DSM", "FAST_CHM", "FAST_TERRAIN",
  "--grid_res", "0.25",
  "--dem_method", "nearest",
  "--dsm_method", "max",
  "--chm_methods", "pitfree",
  "--terrain_products", "all",
  "--apply_fp_fix",
  "--jobs", "8",
  "--joblib_backend", "loky",
  "--overwrite",
  "--overwrite_tiles"
)

cat("Full FAST-GC command:\n")
cat(paste(c(cmd, args_full), collapse = " "))
cat("\n\n")

# Uncomment to run:
# system2(cmd, args = args_full)

# ------------------------------------------------------------------
# Example 3: use a full executable path on Windows
# ------------------------------------------------------------------

# Replace this with your real path if needed
fastgc_exe <- "C:/Users/your_username/anaconda3/envs/fastgc/Scripts/fastgc.exe"

args_windows <- c(
  "--in_path", "F:/lidar_data/USA",
  "--out_dir", "F:/FAST_GC_Test",
  "--sensor_mode", "ALS"
)

cat("Windows explicit executable example:\n")
cat(paste(c(fastgc_exe, args_windows), collapse = " "))
cat("\n\n")

# Uncomment to run:
# system2(fastgc_exe, args = args_windows)

# ------------------------------------------------------------------
# Example 4: capture command output
# ------------------------------------------------------------------

# Uncomment to run and capture logs:
# out <- system2(cmd, args = "--help", stdout = TRUE, stderr = TRUE)
# cat(paste(out, collapse = "\n"))

# ------------------------------------------------------------------
# Notes
# ------------------------------------------------------------------

cat("Notes:\n")
cat("- Edit input and output paths before running.\n")
cat("- Use forward slashes in Windows paths for simplicity.\n")
cat("- If R cannot find fastgc, use the full path to fastgc.exe.\n")
cat("- You can run FAST-GC in Python/conda, then analyze outputs in R.\n")