# ⚠️ Workspace Boundaries & Safety Guidelines

## CRITICAL RESTRICTIONS

### 🚫 Stay Within This Directory
- **NEVER** navigate to or modify files in parent directories outside of `image_stream_analysis`
- Do not access or change anything in:
  - `x:\roy\resources\CODING\` (parent)
  - `x:\roy\resources\` (grandparent)
  - `x:\roy\` (further up)
  - Any system directories

### 🚫 System Files
- **DO NOT** modify any system files
- **DO NOT** change global Python installation
- **DO NOT** modify Windows registry or system settings
- **DO NOT** change system environment variables

### 🚫 Deletion Policy
- **NEVER** delete files outside this directory
- Only delete or modify files within `image_stream_analysis/` when necessary
- Be cautious before deleting anything

### ✅ Python Environment
- Work **ONLY** with a single, isolated Python environment for this project
- Use a virtual environment (venv, conda, or similar) specific to this workspace
- Do **NOT** modify the system Python installation
- Keep all dependencies contained to this project's environment

---

## Guidelines Summary
✓ All work stays within: `x:\roy\resources\CODING\image_stream_analysis\`  
✓ Use an isolated Python environment  
✓ Read-only access to files outside this directory  
✗ No system modifications  
✗ No file deletions outside this directory  

---

**Last Updated:** May 4, 2026

## ImageStream GUI

This project provides a PyQt6 desktop GUI for ImageStream cell analysis. The app is built to:

- load a features table, image folder, and channel map
- match each row in the table to the available cell images
- compute UMAP on selected features
- show feature scatter plots and image previews for the selected cell
- let you search for a cell by Object Number and jump directly to it
- save selected cell images, plots, and AnnData output

## Packaged Windows EXE

The goal of the EXE build is to make the app run on Windows without requiring Python or conda on the machine that opens it.

When the build is complete, the packaged app will:

- launch the same GUI as `run_gui.py`
- include the project `data/` folder needed by the app
- run from the `dist/` folder as a standalone Windows executable

Build helper:

- `build_exe.ps1` packages the GUI with PyInstaller into `dist/`

