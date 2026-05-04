# ImageStream GUI Analysis Tool - Project Plan

## Overview
Build an interactive GUI application for ImageStream cytometry data analysis, combining dimensionality reduction visualization with linked image display.

---

## Feature Requirements

### 1. Data Loading Panel
- **Input 1**: Folder path containing ImageStream images (organized by cell ID)
  - Expected format: Individual image files or multi-channel stacks
- **Input 2**: Channel mapping dictionary (e.g., Channel 1 → "BF", Channel 7 → "DAPI")
  - User specifies which channels to load and their names
- **Input 3**: Features matrix (TXT file)
  - Rows: cells, Columns: feature values (e.g., Area_BF, Intensity_DAPI, etc.)
  - Should be linkable to image filenames by cell ID

### 2. Feature Selection & Clustering
- Display list of available features from the features matrix
- Allow user to **select subset of features** (or "use all features")
- Perform UMAP dimensionality reduction on selected features
- Generate UMAP coordinates for each cell

### 3. Visualization & Interaction (Two Modes)

#### Mode A: UMAP Plot
- Display scatter plot of UMAP coordinates
- Each point = one cell
- **Interactive**: Clicking on a point triggers image display
- Color by feature value (optional user selection)

#### Mode B: Feature Scatter Plot
- User selects **two features** (X and Y axes)
- Create scatter plot of raw feature space
- **Interactive**: Clicking on points triggers image display
- Same linking as UMAP mode

### 4. Image Display Panel (Linked)
When user clicks on a cell in either UMAP or scatter plot:
- **Show 3 sub-panels** for that cell's images:
  1. **BF (Brightfield)** channel alone
  2. **DAPI** channel alone
  3. **DAPI overlaid on BF** with transparency (alpha=0.5)

### 5. Export & Output
- **Output folder selector**: User specifies where to save results
- **Save options**:
  - Export cell image (composite or individual channels as selected)
  - Export AnnData object (.h5ad) containing:
    - Feature matrix
    - UMAP coordinates
    - Cell metadata (image paths, selected features, etc.)

---

## Technical Architecture

### Data Flow
```
Images (folder) + Channel Map + Features Matrix
         ↓
    Load & Align Data
         ↓
    User selects features
         ↓
    Compute UMAP
         ↓
    Display visualization (UMAP or Scatter)
         ↓
    User clicks cell
         ↓
    Load & display linked images
         ↓
    User exports results
```

### Technology Stack
- **GUI Framework**: PyQt6 or Tkinter (user preference?)
- **Data Handling**: AnnData, pandas, numpy
- **Visualization**: matplotlib or plotly (for interactivity)
- **Clustering**: scanpy.tl.umap (via UMAP library)
- **Image I/O**: tifffile, PIL

---

## Confirmed Requirements

1. **Image File Naming**:
   - Format: `{ObjectNumber}_Ch{ChannelNumber}.ome.tif`
   - Example: `0_Ch1.ome.tif`, `0_Ch7.ome.tif`
   - Each cell has one file per channel

2. **Features Matrix Format**:
   - Tab-separated text file
   - First 2 lines: skip (header/metadata)
   - First data column: "Object Number" (links to image filename)
   - Remaining columns: feature values (Area_BF, Intensity_DAPI, etc.)

3. **Channel Mapping**:
   - Read from JSON file: `{"Ch1": "BF", "Ch7": "DAPI", ...}`
   - Maps channel number to display name

4. **GUI Framework**:
   - Fast, modern, interactive plotting with click detection
   - **Decision**: PyQt6 + matplotlib/plotly for responsiveness

5. **UMAP Computation**:
   - **User confirmation required** (button click to compute, not automatic)

6. **Export Format**:
   - Support both **PNG and TIFF** formats for cell images

---

## Next Steps
1. You provide example data from the `data/` folder
2. We clarify the questions above
3. I implement the GUI based on confirmed requirements
