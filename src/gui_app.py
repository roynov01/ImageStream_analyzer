"""PyQt6 GUI application for ImageStream analysis (minimal working version)."""
import sys
from pathlib import Path
from typing import List

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

from .data_loader import load_channel_map, load_features_table, index_images
from .image_io import load_image, overlay_images, normalize_image

import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ImageStream Analysis')
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        # Path fields with browse buttons
        self.features_path = QtWidgets.QLineEdit('data/6_65_allstains_1.txt')
        self.images_path = QtWidgets.QLineEdit(r'C:\Users\royno\Desktop\test\65_imaages\All')
        self.channel_map_path = QtWidgets.QLineEdit('data/channel_map.json')
        self.output_path = QtWidgets.QLineEdit('data/output')

        feat_browse = QtWidgets.QPushButton('Browse')
        feat_browse.clicked.connect(self.browse_features)
        img_browse = QtWidgets.QPushButton('Browse')
        img_browse.clicked.connect(self.browse_images)
        cmap_browse = QtWidgets.QPushButton('Browse')
        cmap_browse.clicked.connect(self.browse_channel_map)
        out_browse = QtWidgets.QPushButton('Browse')
        out_browse.clicked.connect(self.browse_output)

        load_btn = QtWidgets.QPushButton('Load Data')
        load_btn.clicked.connect(self.load_data)

        controls.addWidget(QtWidgets.QLabel('Features:'))
        controls.addWidget(self.features_path)
        controls.addWidget(feat_browse)
        controls.addWidget(QtWidgets.QLabel('Images:'))
        controls.addWidget(self.images_path)
        controls.addWidget(img_browse)
        controls.addWidget(QtWidgets.QLabel('Channel map:'))
        controls.addWidget(self.channel_map_path)
        controls.addWidget(cmap_browse)
        controls.addWidget(QtWidgets.QLabel('Output:'))
        controls.addWidget(self.output_path)
        controls.addWidget(out_browse)
        controls.addWidget(load_btn)

        # Feature selection
        fs_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(fs_layout)
        self.feature_list = QtWidgets.QListWidget()
        self.feature_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        fs_layout.addWidget(self.feature_list)

        umap_btn = QtWidgets.QPushButton('Compute UMAP')
        umap_btn.clicked.connect(self.compute_umap)
        fs_layout.addWidget(umap_btn)

        # Export buttons and selected ID
        self.save_img_btn = QtWidgets.QPushButton('Save Cell Images')
        self.save_img_btn.clicked.connect(self.save_current_cell_images)
        self.save_img_btn.setEnabled(False)
        self.save_adata_btn = QtWidgets.QPushButton('Save AnnData (.h5ad)')
        self.save_adata_btn.clicked.connect(self.save_adata)
        self.save_adata_btn.setEnabled(False)
        fs_layout.addWidget(self.save_img_btn)
        fs_layout.addWidget(self.save_adata_btn)

        self.selected_label = QtWidgets.QLabel('Selected: None')
        fs_layout.addWidget(self.selected_label)

        # Color options
        fs_layout.addWidget(QtWidgets.QLabel('Color feature:'))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.setEnabled(False)
        fs_layout.addWidget(self.color_combo)
        self.p99_chk = QtWidgets.QCheckBox('vmax=p99')
        self.p99_chk.setChecked(True)
        self.p99_chk.setEnabled(False)
        fs_layout.addWidget(self.p99_chk)

        # wire color controls
        self.color_combo.currentIndexChanged.connect(lambda _: self.update_umap_colors())
        self.p99_chk.stateChanged.connect(lambda _: self.update_umap_colors())

        # Matplotlib canvas for plot
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.scatter = None

        # Image display area
        self.img_fig = Figure(figsize=(6, 3))
        self.img_canvas = FigureCanvas(self.img_fig)
        layout.addWidget(self.img_canvas)
        self.img_axes = [self.img_fig.add_subplot(1, 3, i + 1) for i in range(3)]

        # Status / log panel
        self.status_box = QtWidgets.QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(100)
        layout.addWidget(self.status_box)

        # Internal
        self.df = None
        self.adata = None
        self.coords = None
        self.nn = None
        self.selection_marker = None
        self.current_index = None
        self.colorbar = None

    def load_data(self):
        self.log('[LOADING DATA] Starting')
        feats = Path(self.features_path.text())
        imgs = Path(self.images_path.text())
        cmap = Path(self.channel_map_path.text())
        channel_map = load_channel_map(cmap)
        df = load_features_table(feats)
        df = index_images(imgs, df, channel_map)
        self.channel_map = channel_map
        # filter to only rows that have at least one channel path
        path_cols = [str(ch) + '_path' for ch in channel_map.values()]
        available_mask = df[path_cols].notna().any(axis=1)
        df_filtered = df[available_mask].reset_index(drop=True)
        # count available image paths and rows
        found_counts = int(df_filtered[path_cols].notna().sum().sum())
        rows_with_images = int(df_filtered.shape[0])
        self.df = df_filtered
        self.log(f'[LOADED] {len(df)} objects; {rows_with_images} objects have at least one image; total channel paths found: {found_counts}')
        self.feature_list.clear()
        for c in df.columns:
            if c.endswith('_path'):
                continue
            if c == 'Object Number':
                continue
            self.feature_list.addItem(c)
        # do not popup; just log
        # populate color combo with filtered dataframe columns and enable controls
        self.color_combo.clear()
        for c in self.df.columns:
            if c.endswith('_path') or c == 'Object Number':
                continue
            self.color_combo.addItem(c)
        # enable color controls so user can select feature and p99
        self.color_combo.setEnabled(True)
        self.p99_chk.setEnabled(True)
        if self.color_combo.count() > 0:
            self.color_combo.setCurrentIndex(0)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.status_box.append(f'[{ts}] {msg}')

    def browse_features(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select features file', str(Path('.')), 'Text files (*.txt *.tsv);;All files (*)')
        if fn:
            self.features_path.setText(fn)

    def browse_images(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select images folder', str(Path('.')))
        if d:
            self.images_path.setText(d)

    def browse_channel_map(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select channel map JSON', str(Path('.')), 'JSON files (*.json);;All files (*)')
        if fn:
            self.channel_map_path.setText(fn)

    def browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select output folder', str(Path('.')))
        if d:
            self.output_path.setText(d)

    def compute_umap(self):
        if self.df is None:
            self.log('[UMAP] No data loaded')
            return
        items = self.feature_list.selectedItems()
        if not items:
            features = [c for c in self.df.columns if c.endswith('_path') is False and c != 'Object Number']
        else:
            features = [it.text() for it in items]
        X = self.df[features].fillna(0).values
        n_samples, n_features = X.shape
        self.log(f'[UMAP] Computing with {n_samples} samples and {n_features} features')
        try:
            adata = ad.AnnData(X)
            adata.obs = pd.DataFrame(index=range(adata.n_obs))
            if 'Object Number' in self.df.columns:
                adata.obs['Object Number'] = self.df['Object Number'].values
            for col in self.df.columns:
                if col.endswith('_path'):
                    adata.obs[col] = self.df[col].values
            self.display_df = self.df.reset_index().rename(columns={'index': 'orig_index'})
            
            # choose n_comps safely based on adata
            n_samples = int(adata.n_obs)
            n_features = int(adata.n_vars)
            n_comps = min(50, max(1, min(n_samples, n_features)))
            sc.pp.pca(adata, n_comps=n_comps)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            self.adata = adata
            self.coords = adata.obsm['X_umap']
            self.ax.clear()
            self.scatter = self._draw_umap_scatter()
            self.ax.set_title('UMAP')
            self.canvas.draw()
            # build nearest neighbor index
            self.nn = NearestNeighbors(n_neighbors=1).fit(self.coords)
            self.log('[UMAP] Completed successfully')
            # enable save adata
            self.save_adata_btn.setEnabled(True)
        except Exception as e:
            self.log(f'[ERROR] UMAP failed: {e}')
            QtWidgets.QMessageBox.critical(self, 'UMAP error', str(e))

    def _draw_umap_scatter(self):
        if self.coords is None:
            return None
        feat_name = self.color_combo.currentText().strip()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        if feat_name and feat_name in self.display_df.columns:
            vals = pd.to_numeric(self.display_df[feat_name], errors='coerce').fillna(0).values
            vmax = None
            if self.p99_chk.isChecked():
                try:
                    vmax = float(np.nanpercentile(vals, 99))
                except Exception:
                    vmax = None
            scatter = self.ax.scatter(
                self.coords[:, 0],
                self.coords[:, 1],
                c=vals,
                s=20,
                cmap='viridis',
                vmin=0,
                vmax=vmax,
                picker=True,
            )
            try:
                self.colorbar = self.fig.colorbar(scatter, ax=self.ax, label=feat_name)
            except Exception:
                self.colorbar = None
            return scatter
        return self.ax.scatter(self.coords[:, 0], self.coords[:, 1], s=20, c='C0', picker=True)

    def update_umap_colors(self):
        """Update scatter colors without recomputing UMAP."""
        if self.coords is None:
            return
        selected_index = self.current_index
        self.ax.clear()
        self.scatter = self._draw_umap_scatter()
        if selected_index is not None and 0 <= selected_index < len(self.coords):
            sel_x, sel_y = float(self.coords[selected_index, 0]), float(self.coords[selected_index, 1])
            self.selection_marker = self.ax.plot(sel_x, sel_y, marker='o', color='red', markersize=10, markeredgecolor='k', markeredgewidth=1)[0]
        self.ax.set_title('UMAP')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if self.coords is None:
            return
        # find nearest point in UMAP space
        dist, idx = self.nn.kneighbors([[x, y]], return_distance=True)
        i = int(idx[0][0])
        obj_row = self.display_df.iloc[i]

        # mark selected point on UMAP
        sel_x, sel_y = float(self.coords[i, 0]), float(self.coords[i, 1])
        if self.selection_marker is not None:
            try:
                self.selection_marker.remove()
            except Exception:
                pass
        self.selection_marker = self.ax.plot(sel_x, sel_y, marker='o', color='red', markersize=10, markeredgecolor='k', markeredgewidth=1)[0]
        self.canvas.draw()
        self.current_index = i

        # load images with error handling
        bf_path = obj_row.get('BF_path')
        dapi_path = obj_row.get('DAPI_path')
        try:
            bf = load_image(Path(bf_path)) if bf_path else np.zeros((64, 64), dtype=np.float32)
        except Exception as e:
            self.log(f'[ERROR] Failed to read BF image for {obj_row.get("Object Number")}: {e}')
            bf = np.zeros((64, 64), dtype=np.float32)
        try:
            dapi = load_image(Path(dapi_path)) if dapi_path else np.zeros_like(bf)
        except Exception as e:
            self.log(f'[ERROR] Failed to read DAPI image for {obj_row.get("Object Number")}: {e}')
            dapi = np.zeros_like(bf)

        # normalize for display and create overlay
        bf_disp = normalize_image(bf)
        dapi_disp = normalize_image(dapi)
        overlay = overlay_images(bf, dapi)

        # show images
        self.img_axes[0].clear()
        self.img_axes[0].imshow(bf_disp, cmap='gray')
        self.img_axes[0].set_title('BF')
        self.img_axes[1].clear()
        self.img_axes[1].imshow(dapi_disp, cmap='gray')
        self.img_axes[1].set_title('DAPI')
        self.img_axes[2].clear()
        import numpy as _np
        self.img_axes[2].imshow(_np.array(overlay))
        self.img_axes[2].set_title('Overlay')
        for ax in self.img_axes:
            ax.axis('off')
        self.img_canvas.draw()

        # update selected label and enable save image button
        selected_id = obj_row.get('Object Number')
        self.selected_label.setText(f'Selected: {selected_id}')
        self.save_img_btn.setEnabled(True)
        self.log(f'[CLICK] Selected object {selected_id} (index {i})')

    def save_current_cell_images(self):
        if self.current_index is None:
            QtWidgets.QMessageBox.warning(self, 'No selection', 'No cell selected')
            return
        out_dir = Path(self.output_path.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        i = int(self.current_index)
        row = self.display_df.iloc[i]
        obj = str(row.get('Object Number'))
        bf_path = row.get('BF_path')
        dapi_path = row.get('DAPI_path')
        try:
            bf = load_image(Path(bf_path)) if bf_path else np.zeros((64, 64), dtype=np.float32)
        except Exception as e:
            self.log(f'[ERROR] Failed to read BF image for {obj}: {e}')
            bf = np.zeros((64, 64), dtype=np.float32)
        try:
            dapi = load_image(Path(dapi_path)) if dapi_path else np.zeros_like(bf)
        except Exception as e:
            self.log(f'[ERROR] Failed to read DAPI image for {obj}: {e}')
            dapi = np.zeros_like(bf)

        bf_n = (normalize_image(bf) * 255).astype(np.uint8)
        dapi_n = (normalize_image(dapi) * 255).astype(np.uint8)
        from PIL import Image
        bf_img = Image.fromarray(bf_n)
        dapi_img = Image.fromarray(dapi_n)
        overlay_img = overlay_images(bf, dapi)

        # save PNG and TIFF
        bf_png = out_dir / f'{obj}_BF.png'
        bf_tif = out_dir / f'{obj}_BF.tif'
        dapi_png = out_dir / f'{obj}_DAPI.png'
        dapi_tif = out_dir / f'{obj}_DAPI.tif'
        overlay_png = out_dir / f'{obj}_overlay.png'
        overlay_tif = out_dir / f'{obj}_overlay.tif'

        bf_img.save(str(bf_png))
        bf_img.save(str(bf_tif))
        dapi_img.save(str(dapi_png))
        dapi_img.save(str(dapi_tif))
        overlay_img.save(str(overlay_png))
        overlay_img.save(str(overlay_tif))

        self.log(f'[EXPORT] Saved images for {obj} to {out_dir}')
        QtWidgets.QMessageBox.information(self, 'Saved', f'Saved images for {obj} to {out_dir}')

    def save_adata(self):
        if self.adata is None:
            QtWidgets.QMessageBox.warning(self, 'No data', 'No AnnData available')
            return
        out_dir = Path(self.output_path.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'adata.h5ad'
        try:
            self.adata.write_h5ad(str(out_file))
            self.log(f'[EXPORT] Saved AnnData to {out_file}')
            QtWidgets.QMessageBox.information(self, 'Saved', f'Saved AnnData to {out_file}')
        except Exception as e:
            self.log(f'[ERROR] Saving AnnData failed: {e}')
            QtWidgets.QMessageBox.critical(self, 'Save error', str(e))


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()

