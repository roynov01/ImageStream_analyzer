"""PyQt6 GUI application for ImageStream analysis."""
import re
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets
from sklearn.neighbors import NearestNeighbors

from .data_loader import index_images, load_channel_map, load_features_table
from .image_io import load_image, normalize_image, overlay_images, subtract_background


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ImageStream Analysis')
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Data source controls at top
        data_group = QtWidgets.QGroupBox('Data Sources')
        data_grid = QtWidgets.QGridLayout(data_group)
        data_grid.setHorizontalSpacing(6)
        data_grid.setVerticalSpacing(6)
        layout.addWidget(data_group, stretch=0)

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
        load_btn.setMinimumHeight(32)

        data_grid.addWidget(QtWidgets.QLabel('Features:'), 0, 0)
        data_grid.addWidget(self.features_path, 0, 1)
        data_grid.addWidget(feat_browse, 0, 2)
        data_grid.addWidget(load_btn, 0, 3, 2, 1)

        data_grid.addWidget(QtWidgets.QLabel('Images:'), 1, 0)
        data_grid.addWidget(self.images_path, 1, 1)
        data_grid.addWidget(img_browse, 1, 2)

        data_grid.addWidget(QtWidgets.QLabel('Channel map:'), 2, 0)
        data_grid.addWidget(self.channel_map_path, 2, 1)
        data_grid.addWidget(cmap_browse, 2, 2)

        data_grid.addWidget(QtWidgets.QLabel('Output:'), 3, 0)
        data_grid.addWidget(self.output_path, 3, 1)
        data_grid.addWidget(out_browse, 3, 2)

        # Main content split: left (controls + images), right (plots + log)
        body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(body_splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        body_splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        body_splitter.addWidget(right_panel)
        body_splitter.setSizes([520, 900])

        # Controls card
        controls_group = QtWidgets.QGroupBox('Analysis Controls')
        controls_grid = QtWidgets.QGridLayout(controls_group)
        controls_grid.setHorizontalSpacing(6)
        controls_grid.setVerticalSpacing(6)
        controls_grid.setColumnStretch(0, 0)
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(2, 0)
        left_layout.addWidget(controls_group, stretch=0)

        # Feature selection
        self.feature_list = QtWidgets.QListWidget()
        self.feature_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.feature_list.setMinimumHeight(170)
        controls_grid.addWidget(QtWidgets.QLabel('Features for UMAP:'), 0, 0, 1, 3)
        controls_grid.addWidget(self.feature_list, 1, 0, 1, 3)

        umap_btn = QtWidgets.QPushButton('Compute UMAP')
        umap_btn.clicked.connect(self.compute_umap)
        controls_grid.addWidget(umap_btn, 2, 0)

        # Export buttons
        self.save_img_btn = QtWidgets.QPushButton('Save Cell Images')
        self.save_img_btn.clicked.connect(self.save_current_cell_images)
        self.save_img_btn.setEnabled(False)
        self.save_adata_btn = QtWidgets.QPushButton('Save AnnData (.h5ad)')
        self.save_adata_btn.clicked.connect(self.save_adata)
        self.save_adata_btn.setEnabled(False)
        controls_grid.addWidget(self.save_img_btn, 2, 1)
        controls_grid.addWidget(self.save_adata_btn, 2, 2)

        # Color options (off by default)
        self.color_chk = QtWidgets.QCheckBox('Color UMAP by feature')
        self.color_chk.setChecked(False)
        controls_grid.addWidget(self.color_chk, 3, 0, 1, 2)
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.setEnabled(False)
        controls_grid.addWidget(QtWidgets.QLabel('Color feature:'), 4, 0)
        controls_grid.addWidget(self.color_combo, 4, 1, 1, 2)
        self.p99_chk = QtWidgets.QCheckBox('vmax=p99')
        self.p99_chk.setChecked(True)
        self.p99_chk.setEnabled(False)
        controls_grid.addWidget(self.p99_chk, 3, 2)

        controls_grid.addWidget(QtWidgets.QLabel('Feature X:'), 5, 0)
        self.x_feature_combo = QtWidgets.QComboBox()
        self.x_feature_combo.setEnabled(False)
        controls_grid.addWidget(self.x_feature_combo, 5, 1, 1, 2)

        controls_grid.addWidget(QtWidgets.QLabel('Feature Y:'), 6, 0)
        self.y_feature_combo = QtWidgets.QComboBox()
        self.y_feature_combo.setEnabled(False)
        controls_grid.addWidget(self.y_feature_combo, 6, 1, 1, 2)

        controls_grid.addWidget(QtWidgets.QLabel('DAPI gain:'), 7, 0)
        self.dapi_gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dapi_gain_slider.setMinimum(10)
        self.dapi_gain_slider.setMaximum(300)
        self.dapi_gain_slider.setValue(100)
        self.dapi_gain_slider.setTickInterval(10)
        controls_grid.addWidget(self.dapi_gain_slider, 7, 1)
        self.dapi_gain_label = QtWidgets.QLabel('1.00x')
        controls_grid.addWidget(self.dapi_gain_label, 7, 2)

        # wire color controls
        self.color_chk.stateChanged.connect(self._toggle_color_controls)
        self.color_combo.currentIndexChanged.connect(lambda _: self.update_umap_colors())
        self.p99_chk.stateChanged.connect(lambda _: self.update_umap_colors())
        self.x_feature_combo.currentIndexChanged.connect(lambda _: self.update_feature_scatter())
        self.y_feature_combo.currentIndexChanged.connect(lambda _: self.update_feature_scatter())
        self.dapi_gain_slider.valueChanged.connect(self._on_dapi_gain_changed)

        # Image panel: 1 column x 3 rows
        image_group = QtWidgets.QGroupBox('Selected Cell Images')
        image_layout = QtWidgets.QVBoxLayout(image_group)
        left_layout.addWidget(image_group, stretch=1)
        self.img_fig = Figure(figsize=(5, 8))
        self.img_canvas = FigureCanvas(self.img_fig)
        image_layout.addWidget(self.img_canvas)
        self.img_axes = [self.img_fig.add_subplot(3, 1, i + 1) for i in range(3)]

        # Plot panel
        plots_group = QtWidgets.QGroupBox('UMAP and Feature Scatter')
        plots_layout = QtWidgets.QVBoxLayout(plots_group)
        right_layout.addWidget(plots_group, stretch=1)
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        plots_layout.addWidget(self.canvas)
        self.ax_umap = self.fig.add_subplot(1, 2, 1)
        self.ax_feat = self.fig.add_subplot(1, 2, 2)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Status / log panel (right side only)
        status_group = QtWidgets.QGroupBox('Status Log')
        status_layout = QtWidgets.QVBoxLayout(status_group)
        right_layout.addWidget(status_group, stretch=0)
        self.status_box = QtWidgets.QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(120)
        status_layout.addWidget(self.status_box)

        # Internal
        self.df = None
        self.adata = None
        self.coords = None
        self.feat_coords = None
        self.nn_umap = None
        self.nn_feat = None
        self.selection_marker_umap = None
        self.selection_marker_feat = None
        self.current_index = None
        self.umap_scatter = None
        self.feat_scatter = None
        self.colorbar = None
        self.display_df = None
        self.dapi_scale = None

    def load_data(self):
        self.log('[LOADING DATA] Starting')
        feats = Path(self.features_path.text())
        imgs = Path(self.images_path.text())
        cmap = Path(self.channel_map_path.text())
        channel_map = load_channel_map(cmap)
        df = load_features_table(feats)
        df = index_images(imgs, df, channel_map)
        df = self._rename_feature_columns(df)
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
        self._populate_feature_controls()
        self._estimate_dapi_scale()

    def _rename_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        used = set(df.columns)
        for col in list(df.columns):
            if col == 'Object Number' or col.endswith('_path'):
                continue
            new_col = col
            new_col = new_col.replace('M07_DAPI', 'DAPI')
            new_col = new_col.replace('_M1', '_BF')
            new_col = new_col.replace(' M07 DAPI', ' DAPI')
            new_col = re.sub(r'\s+', ' ', new_col).strip()
            if new_col != col:
                candidate = new_col
                suffix = 2
                while candidate in used and candidate != col:
                    candidate = f'{new_col}_{suffix}'
                    suffix += 1
                rename_map[col] = candidate
                used.add(candidate)
        if rename_map:
            df = df.rename(columns=rename_map)
            self.log(f'[LOAD] Renamed {len(rename_map)} feature columns (_M1->BF, M07_DAPI->DAPI)')
        return df

    def _populate_feature_controls(self):
        self.feature_list.clear()
        self.color_combo.clear()
        self.x_feature_combo.clear()
        self.y_feature_combo.clear()

        feature_cols = [c for c in self.df.columns if c != 'Object Number' and not c.endswith('_path')]
        for col in feature_cols:
            self.feature_list.addItem(col)
            self.color_combo.addItem(col)
            self.x_feature_combo.addItem(col)
            self.y_feature_combo.addItem(col)

        self.color_combo.setEnabled(False)
        self.p99_chk.setEnabled(False)
        self.x_feature_combo.setEnabled(True)
        self.y_feature_combo.setEnabled(True)

        # default feature scatter: Area BF vs DAPI intensity when available
        x_default = self._find_feature_name(['Area', 'BF'])
        y_default = self._find_feature_name(['DAPI', 'Intensity'])
        if x_default is not None:
            self.x_feature_combo.setCurrentText(x_default)
        if y_default is not None:
            self.y_feature_combo.setCurrentText(y_default)

    def _find_feature_name(self, tokens):
        token_set = [t.lower() for t in tokens]
        for i in range(self.x_feature_combo.count()):
            name = self.x_feature_combo.itemText(i)
            low = name.lower()
            if all(t in low for t in token_set):
                return name
        return self.x_feature_combo.itemText(0) if self.x_feature_combo.count() > 0 else None

    def _estimate_dapi_scale(self):
        self.dapi_scale = None
        if self.df is None or 'DAPI_path' not in self.df.columns:
            return
        dapi_paths = self.df['DAPI_path'].dropna()
        if dapi_paths.empty:
            return
        sample_count = min(150, len(dapi_paths))
        sample_paths = dapi_paths.sample(n=sample_count, random_state=42)
        vals = []
        for p in sample_paths:
            try:
                img = load_image(Path(p))
                img_bg = subtract_background(img)
                vals.append(float(np.nanpercentile(img_bg, 99.5)))
            except Exception:
                continue
        if vals:
            self.dapi_scale = max(float(np.median(vals)), 1.0)
            self.log(f'[LOAD] DAPI fixed scale set to {self.dapi_scale:.2f} (p99.5 median from {len(vals)} sampled cells)')

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

    def _toggle_color_controls(self, state):
        enabled = int(state) == int(QtCore.Qt.CheckState.Checked.value)
        self.color_combo.setEnabled(enabled)
        self.p99_chk.setEnabled(enabled)
        self.update_umap_colors()

    def _on_dapi_gain_changed(self, value):
        gain = value / 100.0
        self.dapi_gain_label.setText(f'{gain:.2f}x')
        if self.current_index is not None:
            self._show_selected_images(self.current_index)

    def compute_umap(self):
        if self.df is None:
            self.log('[UMAP] No data loaded')
            return
        items = self.feature_list.selectedItems()
        if not items:
            features = [c for c in self.df.columns if c.endswith('_path') is False and c != 'Object Number']
        else:
            features = [it.text() for it in items]
        X = self.df[features].apply(pd.to_numeric, errors='coerce').fillna(0).values
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
            self.display_df = self.df.reset_index(drop=True)
            
            # choose n_comps safely based on adata
            n_samples = int(adata.n_obs)
            n_features = int(adata.n_vars)
            n_comps = min(50, max(1, min(n_samples, n_features)))
            sc.pp.pca(adata, n_comps=n_comps)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            self.adata = adata
            self.coords = adata.obsm['X_umap']
            self.nn_umap = NearestNeighbors(n_neighbors=1).fit(self.coords)

            self._update_feature_coords()
            self._redraw_plots()
            self.log('[UMAP] Completed successfully')
            # enable save adata
            self.save_adata_btn.setEnabled(True)
        except Exception as e:
            self.log(f'[ERROR] UMAP failed: {e}')
            QtWidgets.QMessageBox.critical(self, 'UMAP error', str(e))

    def _update_feature_coords(self):
        if self.display_df is None:
            return
        x_name = self.x_feature_combo.currentText().strip()
        y_name = self.y_feature_combo.currentText().strip()
        if x_name not in self.display_df.columns or y_name not in self.display_df.columns:
            self.feat_coords = None
            self.nn_feat = None
            return
        xvals = pd.to_numeric(self.display_df[x_name], errors='coerce').fillna(0).values.astype(np.float32)
        yvals = pd.to_numeric(self.display_df[y_name], errors='coerce').fillna(0).values.astype(np.float32)
        self.feat_coords = np.column_stack([xvals, yvals])
        self.nn_feat = NearestNeighbors(n_neighbors=1).fit(self.feat_coords)

    def _draw_umap_scatter(self):
        if self.coords is None:
            return None

        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

        if self.color_chk.isChecked():
            feat_name = self.color_combo.currentText().strip()
            if feat_name and feat_name in self.display_df.columns:
                vals = pd.to_numeric(self.display_df[feat_name], errors='coerce').fillna(0).values
                vmax = None
                if self.p99_chk.isChecked():
                    try:
                        vmax = float(np.nanpercentile(vals, 99))
                    except Exception:
                        vmax = None
                sca = self.ax_umap.scatter(
                    self.coords[:, 0],
                    self.coords[:, 1],
                    c=vals,
                    s=16,
                    cmap='viridis',
                    vmin=0,
                    vmax=vmax,
                    picker=True,
                )
                try:
                    self.colorbar = self.fig.colorbar(sca, ax=self.ax_umap, label=feat_name)
                except Exception:
                    self.colorbar = None
                return sca

        return self.ax_umap.scatter(self.coords[:, 0], self.coords[:, 1], s=16, c='C0', picker=True)

    def _draw_feature_scatter(self):
        if self.feat_coords is None:
            return None
        x_name = self.x_feature_combo.currentText().strip()
        y_name = self.y_feature_combo.currentText().strip()
        sca = self.ax_feat.scatter(self.feat_coords[:, 0], self.feat_coords[:, 1], s=14, c='0.35', alpha=0.9, picker=True)
        self.ax_feat.set_xlabel(x_name)
        self.ax_feat.set_ylabel(y_name)
        self.ax_feat.set_title(f'{x_name} vs {y_name}')
        return sca

    def _redraw_plots(self):
        if self.coords is None:
            return
        self.ax_umap.clear()
        self.ax_feat.clear()
        self.umap_scatter = self._draw_umap_scatter()
        self.feat_scatter = self._draw_feature_scatter()
        self.ax_umap.set_title('UMAP')
        self.fig.tight_layout(pad=1.0)

        if self.current_index is not None:
            self._draw_selection_markers()
        self.canvas.draw()

    def _draw_selection_markers(self):
        if self.current_index is None:
            return
        i = int(self.current_index)

        if self.selection_marker_umap is not None:
            try:
                self.selection_marker_umap.remove()
            except Exception:
                pass
        if self.selection_marker_feat is not None:
            try:
                self.selection_marker_feat.remove()
            except Exception:
                pass

        if self.coords is not None and 0 <= i < len(self.coords):
            self.selection_marker_umap = self.ax_umap.plot(
                float(self.coords[i, 0]),
                float(self.coords[i, 1]),
                marker='o',
                color='red',
                markersize=9,
                markeredgecolor='k',
                markeredgewidth=1,
            )[0]
        if self.feat_coords is not None and 0 <= i < len(self.feat_coords):
            self.selection_marker_feat = self.ax_feat.plot(
                float(self.feat_coords[i, 0]),
                float(self.feat_coords[i, 1]),
                marker='o',
                color='red',
                markersize=9,
                markeredgecolor='k',
                markeredgewidth=1,
            )[0]

    def update_umap_colors(self):
        """Update scatter colors without recomputing UMAP."""
        if self.coords is None:
            return
        self._redraw_plots()

    def update_feature_scatter(self):
        if self.coords is None or self.display_df is None:
            return
        self._update_feature_coords()
        self._redraw_plots()

    def on_click(self, event):
        if event.inaxes not in (self.ax_umap, self.ax_feat):
            return
        x, y = event.xdata, event.ydata
        if self.coords is None or x is None or y is None:
            return
        if event.inaxes == self.ax_umap and self.nn_umap is not None:
            _, idx = self.nn_umap.kneighbors([[x, y]], return_distance=True)
            self._select_index(int(idx[0][0]))
        elif event.inaxes == self.ax_feat and self.nn_feat is not None:
            _, idx = self.nn_feat.kneighbors([[x, y]], return_distance=True)
            self._select_index(int(idx[0][0]))

    def _select_index(self, i: int):
        self.current_index = int(i)
        self._draw_selection_markers()
        self.canvas.draw()
        self._show_selected_images(self.current_index)
        self.save_img_btn.setEnabled(True)

    def _show_selected_images(self, i: int):
        if self.display_df is None or i < 0 or i >= len(self.display_df):
            return
        obj_row = self.display_df.iloc[i]

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

        # BF is always shown fully as the base channel.
        bf_disp = normalize_image(bf)

        # DAPI uses background subtraction and fixed normalization for cross-cell comparison.
        dapi_gain = self.dapi_gain_slider.value() / 100.0
        dapi_bg = subtract_background(dapi)
        scale = self.dapi_scale if self.dapi_scale is not None else max(float(np.nanpercentile(dapi_bg, 99.5)), 1.0)
        dapi_disp = np.clip((dapi_bg / scale) * dapi_gain, 0.0, 1.0)

        overlay = overlay_images(bf, dapi, alpha=0.85, dapi_scale=scale, dapi_gain=dapi_gain)

        self.img_axes[0].clear()
        self.img_axes[0].imshow(bf_disp, cmap='gray')
        self.img_axes[0].set_title('BF')
        self.img_axes[1].clear()
        self.img_axes[1].imshow(dapi_disp, cmap='gray')
        self.img_axes[1].set_title(f'DAPI ({dapi_gain:.2f}x)')
        self.img_axes[2].clear()
        self.img_axes[2].imshow(np.array(overlay))
        self.img_axes[2].set_title('Overlay')
        for ax in self.img_axes:
            ax.axis('off')
        self.img_fig.tight_layout(pad=0.8)
        self.img_canvas.draw()

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
        dapi_bg = subtract_background(dapi)
        dapi_gain = self.dapi_gain_slider.value() / 100.0
        scale = self.dapi_scale if self.dapi_scale is not None else max(float(np.nanpercentile(dapi_bg, 99.5)), 1.0)
        dapi_n = (np.clip((dapi_bg / scale) * dapi_gain, 0.0, 1.0) * 255).astype(np.uint8)
        from PIL import Image
        bf_img = Image.fromarray(bf_n)
        dapi_img = Image.fromarray(dapi_n)
        overlay_img = overlay_images(bf, dapi, alpha=0.85, dapi_scale=scale, dapi_gain=dapi_gain)

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

