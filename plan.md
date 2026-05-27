# ImageStream GUI Analysis Tool - Project Plan (Updated)

## Summary
Interactive PyQt6 application that loads ImageStream images plus a tab-separated features table, computes UMAP from selectable features, and links interactive UMAP / feature-scatter plots to per-cell images (BF, DAPI, overlay). Supports export of per-cell images and an AnnData (.h5ad) session file including UMAP coords and display settings.

## Core Functionality (current)
- Data inputs:
     - Images folder (files named like `{ObjectNumber}_Ch{ChannelKey}.ome.tif` or `.tif`)
     - Features TSV (first data column -> `Object Number`)
     - Channel map JSON mapping channel keys (e.g. `Ch1`) -> friendly names (e.g. `BF`, `DAPI`)
- Robust loading:
     - `index_images()` searches multiple filename patterns and zero-padding variants.
     - Feature table loader normalizes `Object Number` column.
- Feature processing:
     - Column renaming using channel aliases (BF, DAPI, etc.)
     - Automatic area and length/width ratio feature creation when applicable
     - Option to filter features based on available channels
- Dimensionality reduction:
     - User selects features (or all); click "Compute UMAP" to run
     - Uses AnnData + scaled layers, PCA with safe `n_comps` logic, neighbor graph, UMAP
     - Handles small-n edge cases (skips if insufficient samples/features)
- Visualization:
     - UMAP scatter (click selects nearest cell)
     - Feature scatter (choose X and Y features; log options)
     - Color by feature with optional p99 clipping and scaled/raw toggle
     - Selection markers synchronize between plots
- Linked image panel:
     - Shows 3 panes: BF, DAPI (background-subtracted and normalized), and overlay (BF + cyan DAPI)
     - Lazy image caching with limit (default 50 images)
     - Per-image pixel-size detection from OME metadata (draws 10 µm scalebar if available)
     - Image export (PNG) for selected cell; overlay produced with alpha blending
- Persistence / export:
     - Save current session as AnnData `.h5ad` including `adata.uns['session_settings']` for display restoration
     - Save plots and individual cell images to chosen output folder

## UI Controls (current)
- Paths: Features, Images, Channel map, Output
- Buttons: `Load Data`, `Load H5AD`, `RESET`, `Compute UMAP`, `Save Cell Images`, `Save AnnData`, `Save Plots`
- Feature controls: multi-select feature list for UMAP; X/Y feature selectors for scatter
- Visualization toggles: color by feature, p99 clipping, use scaled features, log axes, DAPI min/max sliders
- Scalebar toggle (enabled when pixel-size is detected)

## File formats & naming (confirmed)
- Features: tab-separated text; first data column must map to filenames via `Object Number`
- Channel map: JSON `{"Ch1":"BF","Ch7":"DAPI",...}`
- Image naming expected: `{ObjectNumber}_{ChannelKey}.ome.tif` (indexer tries variants)

## Developer notes / gotchas
- Run as a module or via a small launcher to preserve relative imports.
- `load_image()` uses `tifffile`, `scikit-image` fallback, and PIL as last resort to handle various OME-TIFF variants.
- DAPI scaling: app estimates a fixed dapi_scale from sampled cells (p99.5 mean), used for consistent display across cells.
- UMAP is not auto-run on load — explicit user action avoids unexpected CPU work.

## Short roadmap (suggested)
- Add a headless CLI mode to compute UMAP & export AnnData without launching GUI.
- Add configurable cache size and asynchronous image loading for very large datasets.
- Add unit tests for `index_images`, `overlay_images`, and `get_pixel_size_um`.



