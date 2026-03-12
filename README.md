# FAST-GC <img src="images/logo_fastgc.png" align="right" width="180"/>

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-research%20software-green)

## Fully Adaptive Self-Tuning Sensor-Agnostic Ground Classification for LiDAR Point Clouds

FAST-GC (Fully Adaptive Self-Tuning Sensor-Agnostic Ground Classification) is a robust LiDAR processing framework designed for **automated ground classification and terrain modeling** across multiple LiDAR acquisition systems.

The algorithm is designed to be:

- **Parameter-free**
- **Sensor-agnostic**
- **Computationally scalable**
- **Suitable for large LiDAR datasets**

FAST-GC automatically adapts its processing pipeline according to sensor modality and point cloud characteristics, enabling consistent ground classification across diverse environments and survey configurations.

FAST-GC currently supports:

- ground classification
- optional false-positive correction (FP-Fix)
- digital elevation model generation
- digital surface model generation
- canopy height model generation
- DEM-normalized LiDAR point clouds
- automatic tiling and tile merging for large datasets

---

# Article Title

## FAST-GC: Fully Adaptive Self-Tuning Sensor-Agnostic Ground Classification Algorithm for LiDAR Point Clouds

**Authors**  
Nadeem Fareed et al.

**Manuscript status**  
Manuscript in preparation.

---

# Supported LiDAR Sensors

FAST-GC supports multiple LiDAR acquisition systems.

| Sensor | Description |
|--------|-------------|
| ALS | Airborne Laser Scanning |
| ULS | UAV Laser Scanning |
| TLS | Terrestrial Laser Scanning |

The algorithm automatically adapts to sensor geometry, point density, and acquisition characteristics.

---

# Why FAST-GC?

Ground classification is one of the most critical preprocessing steps in LiDAR analysis because it directly affects the quality of downstream terrain and canopy products.

FAST-GC is designed to provide a unified workflow for:

- robust ground / non-ground separation
- scalable processing of large LiDAR datasets
- cross-sensor support for ALS, ULS, and TLS data
- terrain derivative generation from classified outputs

Unlike many traditional workflows that require significant manual parameter tuning, FAST-GC is designed to operate in a **self-tuning and sensor-adaptive** manner.

---

# Key Features

- Sensor-agnostic ground classification
- Parameter-free processing pipeline
- Built-in False Positive correction (FP-Fix)
- Automatic tiling for massive datasets
- Batch processing support
- Tile merging for seamless final outputs
- Terrain derivative generation
- Optional post-processing of derived raster products
- Support for large wall-to-wall LiDAR processing

**Outputs include:**

| Output | Description |
|--------|-------------|
| `FAST_GC` | Classified LiDAR point cloud |
| `FAST_DEM` | Digital Elevation Model |
| `FAST_DSM` | Digital Surface Model |
| `FAST_CHM` | Canopy Height Model |
| `FAST_NORMALIZED` | DEM-normalized LiDAR point cloud |

---

# FAST-GC Processing Workflow

```text
Input LiDAR
↓
Automatic Tiling (optional)
↓
Ground Classification
↓
FP-Fix Correction (optional)
↓
Final Ground Points
↓
DEM / DSM / CHM / Normalized
↓
Merge Tiles
Example Outputs

The following placeholders assume all figures are stored in a single images/ folder at the repository root.

FAST-GC Ground Classification
Original LiDAR Point Cloud	FAST-GC Ground Classification

	

Color legend:

Green → Ground points

Blue → Non-ground points

FAST-GC DEM

FAST-GC CHM

FAST-GC Normalized Point Cloud

FAST-GC FP-Fix

FAST-GC Hillshade

Installation

FAST-GC can be installed using pip or conda.

Install with pip
pip install fastgc
Install from source
git clone https://github.com/nadeemfareed/FAST-GC.git
cd FAST-GC
pip install -e .
Conda environment (recommended)
conda create -n fastgc python=3.10
conda activate fastgc
pip install fastgc
Command-Line Interface

FAST-GC provides a command-line interface through the fastgc command.

Basic syntax:

fastgc --in_path <input> --sensor_mode <ALS|ULS|TLS>

Typical arguments include:

--in_path : input LAS/LAZ file or folder

--out_dir : output directory

--sensor_mode : ALS / ULS / TLS

--workflow : processing workflow

--products : requested output products

--grid_res : raster resolution

--tile_size_m : tile size in meters

--buffer_m : tile buffer size

--no_fp_fix : disable FP-Fix

--recursive : recursively scan folders

Pre-Processing (Tiling Large LiDAR Files)

Large LiDAR datasets should be tiled for efficient processing.

Important Parameters
Parameter	Description
tile_size_m	Size of processing tiles
buffer_m	Overlap between tiles
sensor_mode	ALS / TLS / ULS
small_tile_merge_frac	Merge tiles smaller than threshold
overwrite_tiles	Force rebuild tiles
Example — Tiling Only
fastgc \
  --in_path "F:\lidar_data\USA" \
  --out_dir "F:\lidar_data" \
  --sensor_mode ALS \
  --workflow tile-only \
  --tile_size_m 100 \
  --buffer_m 5 \
  --recursive

Output structure:

ALS_tiles/
   tiles/
   tile_manifest.json
Ground Classification

FAST-GC performs multi-stage ground classification.

Steps

Initial ground detection

DEM construction

Residual analysis

FP-Fix correction (optional)

Final ground classification

Run FAST-GC on a Single File
fastgc \
  --in_path input.las \
  --sensor_mode ALS \
  --products FAST_GC
Batch Processing (Tiled Dataset)
fastgc \
  --in_path "F:\lidar_data\ALS_tiles\tiles" \
  --out_dir "F:\lidar_data\ALS_tiles" \
  --sensor_mode ALS \
  --workflow run \
  --products FAST_GC \
  --recursive
Disable FP-Fix (optional)
--no_fp_fix
FP-Fix Logic

FP-Fix is an optional post-classification correction step that uses DEM-normalized residuals to reduce classification leakage.

Concept

A temporary DEM is generated from ground points. Each point is then compared to the DEM surface using:

residual = Zpoint − Zdem

The residual is used to detect likely false positives and false negatives in the classification.

Rules
Condition	Action
Non-ground point with normalized elevation ≤ 0	Convert to ground
Ground point with normalized elevation > 6 cm	Convert to non-ground

Temporary surfaces used for FP-Fix:

provisional FAST_DEM
provisional FAST_NORMALIZED

These are removed automatically unless:

--keep_fp_fix_temp
Terrain Derivative Products

FAST-GC can generate terrain products directly.

Product	Description
FAST_DEM	Digital Elevation Model
FAST_DSM	Digital Surface Model
FAST_CHM	Canopy Height Model
FAST_NORMALIZED	DEM-normalized point cloud

Outputs are provided in:

LAS format

GeoTIFF raster format

Raster Creation Methods
DEM Methods
Method	Description
min	Minimum Z value
max	Maximum Z value
mean	Average
nearest	Nearest point
idw	Inverse Distance Weighting

Default:

--dem_method min
DSM Methods
Method	Description
min	Minimum Z value
max	Maximum Z value
mean	Average
nearest	Nearest point
idw	Inverse Distance Weighting

Default:

--dsm_method max
CHM Noise Filtering

Small canopy noise (salt-and-pepper artifacts) may occur due to isolated high points.

FAST-GC provides optional CHM smoothing.

Parameter	Description
chm_median_size	Median filter window
chm_min_height	Minimum canopy threshold

Example:

--chm_median_size 3 \
--chm_min_height 0.25
Example — Generate All Products
fastgc \
  --in_path "F:\lidar_data\ALS_tiles\tiles" \
  --out_dir "F:\lidar_data\ALS_tiles" \
  --sensor_mode ALS \
  --workflow run \
  --products all \
  --grid_res 0.25 \
  --recursive
Example — Generate Selected Products
DEM only
fastgc --in_path input.las --sensor_mode ALS --products FAST_DEM
DSM only
fastgc --in_path input.las --sensor_mode ALS --products FAST_DSM
CHM only
fastgc --in_path input.las --sensor_mode ALS --products FAST_CHM
DEM + DSM
fastgc --in_path input.las --sensor_mode ALS --products FAST_DEM FAST_DSM
Example — Derive Missing Products from Existing FAST_GC

If FAST_GC already exists, FAST-GC can derive downstream products without re-running ground classification.

Example — derive CHM only
fastgc \
  --in_path "F:\lidar_data\Utah\ALS_tiles" \
  --sensor_mode ALS \
  --workflow derive-only \
  --products FAST_CHM \
  --grid_res 0.5
Example — derive DEM, NORMALIZED, and CHM
fastgc \
  --in_path "F:\lidar_data\Utah\ALS_tiles" \
  --sensor_mode ALS \
  --workflow derive-only \
  --products FAST_DEM FAST_NORMALIZED FAST_CHM \
  --grid_res 0.5

This workflow is especially useful when:

FAST_GC has already been completed

FP-Fix has already been applied

one or more downstream products need to be rebuilt

long reprocessing of classification should be avoided

Tile Merge

When processing tiled datasets, tiles must be merged into final outputs.

fastgc \
  --in_path "F:\lidar_data\ALS_tiles" \
  --sensor_mode ALS \
  --workflow merge

Final outputs:

Merged_ALS/

FAST_GC.las
FAST_DEM.tif
FAST_DSM.tif
FAST_CHM.tif
FAST_NORMALIZED.las
Complete Example (Single Large LAS File)
fastgc \
  --in_path "F:\lidar_data\Utah_201517.laz" \
  --out_dir "F:\lidar_data" \
  --sensor_mode ALS \
  --workflow tile-run-merge \
  --products all \
  --tile_size_m 250 \
  --buffer_m 5 \
  --grid_res 0.25

Pipeline executed:

Tile dataset

Ground classification

FP-Fix correction (optional)

Point cloud normalization

DEM generation

DSM generation

CHM generation

Tile merging

Output Structure

A typical processed workspace looks like this:

FAST-GC/
│
├── images/
│   ├── original_pointcloud.png
│   ├── fastgc_ground.png
│   ├── fastgc_dem.png
│   ├── fastgc_chm.png
│   ├── fastgc_normalized.png
│   ├── fastgc_fpfix.png
│   └── fastgc_hillshade.png
│
├── README.md
└── LICENSE

A typical FAST-GC output folder may contain:

Processed_ALS/
│
├── FAST_GC/
├── FAST_DEM/
├── FAST_DSM/
├── FAST_CHM/
└── FAST_NORMALIZED/

For tiled workflows:

ALS_tiles/
│
├── tiles/
├── tile_manifest.json
├── Processed_ALS/
└── Merged_ALS/
Citation

If you use FAST-GC in research please cite:

FAST-GC: Fully Adaptive Self-Tuning Sensor-Agnostic Ground Classification Algorithm for LiDAR Point Clouds

Authors
Nadeem Fareed et al.

License

FAST-GC is released under the Apache License 2.0. See the LICENSE
 file for details.

Acknowledgements

FAST-GC development builds upon advances in:

LiDAR terrain analysis

point cloud processing

open-source geospatial computing

About

FAST-GC is a research software framework for scalable LiDAR ground classification and terrain modeling across ALS, ULS, and TLS systems.

It is intended for applications in:

terrain modeling

forestry

vegetation structure analysis

wall-to-wall LiDAR processing

large-scale geospatial workflows