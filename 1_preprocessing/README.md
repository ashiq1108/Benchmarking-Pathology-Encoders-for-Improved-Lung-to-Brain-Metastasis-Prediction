# Stage 1 — WSI Preprocessing

**Notebook:** `tcia_preprocessing.ipynb`  
**Input:** Raw `.svs` NSCLC slides from TCIA  
**Output:** One `.h5` file per slide containing tiles + coordinates

## What this notebook does

1. Loads each `.svs` file using OpenSlide at 20× effective magnification
2. Runs Otsu thresholding to build a tissue mask — discards background tiles
3. Applies a Laplacian blur filter to drop out-of-focus tiles (`MIN_LAP_VAR = 100`)
4. Fits a Vahadane (SPCN) stain normalizer on the Zhou et al. reference tile
5. Normalizes every kept tile to the reference color space
6. Saves tiles + (x, y) coordinates into a `.h5` file per slide
7. Checkpointing: skips slides whose `.h5` already exists — safe to resume

## Configuration (Cell 1)

| Parameter | Value | Notes |
|---|---|---|
| `TILE_SIZE` | 256 px | At 20× magnification |
| `TISSUE_THRESHOLD` | 0.80 | Min tissue fraction per tile |
| `MIN_LAP_VAR` | 100.0 | Blur rejection threshold |
| `SVS_DIR` | your path | Folder containing `.svs` files |
| `OUTPUT_DIR` | your path | Where `.h5` files are saved |
| `REF_TILE` | your path | Zhou et al. reference tile (`.png`) |

## Run order

Cell 1 → 2 → 3 → 4 (reference check only) → 5 → 6 → 7 → 8 → 9 (sanity check)

## Expected output

- ~388,000 tiles across 96 TCIA slides
- Processing time: ~3–8 min per slide
- One `.h5` per slide with keys `tiles` (uint8) and `coords` (int32)

## Dependencies

Run on **local machine** (needs OpenSlide system library):
```bash
# Ubuntu
sudo apt-get install openslide-tools
pip install openslide-python staintools spams-bin h5py
```
