"""PyQt6 GUI application for ImageStream analysis."""
import argparse
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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties

from .data_loader import index_images, load_channel_map, load_features_table
from .image_io import get_pixel_size_um, load_image, normalize_image, overlay_images, subtract_background


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='ImageStream analysis GUI')
    parser.add_argument('--images', dest='images_path', default=r'C:\Users\royno\Desktop\test', help='Folder containing image files')
    parser.add_argument('--features', dest='features_path', default=r'C:\Users\royno\Desktop\test\6_65_allstains_1.txt', help='Features TXT/TSV file')
    parser.add_argument('--channel-map', dest='channel_map_path', default='data/channel_map.json', help='Channel map JSON file')
    parser.add_argument('--output', dest='output_path', default='data/output', help='Output folder')
    parser.add_argument('--name-prefix', dest='name_prefix', default='', help='Prefix to use for exported files')
    parser.add_argument('--x-axis', dest='x_feature', default='', help='Default feature for the X axis scatter plot')
    parser.add_argument('--y-axis', dest='y_feature', default='', help='Default feature for the Y axis scatter plot')
    parser.add_argument('--autoload', dest='autoload', action='store_true', help='Load the selected files immediately when the GUI opens')
    parser.add_argument('--no-autoload', dest='autoload', action='store_false', help='Open the GUI without loading data')
    parser.set_defaults(autoload=False)
    return parser.parse_args(argv)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, startup_config=None):
        super().__init__()
        self.startup_config = startup_config or {}
        self.startup_x_feature = self.startup_config.get('x_feature', '').strip()
        self.startup_y_feature = self.startup_config.get('y_feature', '').strip()
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
        self.features_path = QtWidgets.QLineEdit(r'C:\Users\royno\Desktop\test\6_65_allstains_1.txt')
        self.images_path = QtWidgets.QLineEdit(r'C:\Users\royno\Desktop\test')
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
        load_h5ad_btn = QtWidgets.QPushButton('Load H5AD')
        load_h5ad_btn.clicked.connect(self.load_adata_file)

        # Vertical button column: Load Data on top, Load H5AD below
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(6)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(load_h5ad_btn)

        data_grid.addWidget(QtWidgets.QLabel('Features:'), 0, 0)
        data_grid.addWidget(self.features_path, 0, 1)
        data_grid.addWidget(feat_browse, 0, 2)
        data_grid.addLayout(btn_layout, 0, 3, 5, 1)

        data_grid.addWidget(QtWidgets.QLabel('Images:'), 1, 0)
        data_grid.addWidget(self.images_path, 1, 1)
        data_grid.addWidget(img_browse, 1, 2)

        data_grid.addWidget(QtWidgets.QLabel('Channel map:'), 2, 0)
        data_grid.addWidget(self.channel_map_path, 2, 1)
        data_grid.addWidget(cmap_browse, 2, 2)

        data_grid.addWidget(QtWidgets.QLabel('Output:'), 3, 0)
        data_grid.addWidget(self.output_path, 3, 1)
        data_grid.addWidget(out_browse, 3, 2)

        data_grid.addWidget(QtWidgets.QLabel('Name prefix:'), 4, 0)
        self.name_prefix = QtWidgets.QLineEdit('')
        self.name_prefix.setPlaceholderText('optional')
        data_grid.addWidget(self.name_prefix, 4, 1, 1, 2)

        self._apply_startup_config()

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

        # Analysis controls: only UMAP features + compute button
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
        umap_btn.setMinimumHeight(34)
        umap_btn.setMinimumWidth(220)
        controls_grid.addWidget(umap_btn, 2, 0, 1, 3)

        # Visualization controls
        viz_group = QtWidgets.QGroupBox('Visualization Controls')
        viz_grid = QtWidgets.QGridLayout(viz_group)
        viz_grid.setHorizontalSpacing(6)
        viz_grid.setVerticalSpacing(6)
        viz_grid.setColumnStretch(0, 0)
        viz_grid.setColumnStretch(1, 1)
        viz_grid.setColumnStretch(2, 0)
        left_layout.addWidget(viz_group, stretch=0)

        # Color options (on by default)
        self.color_chk = QtWidgets.QCheckBox('Color UMAP by feature')
        self.color_chk.setChecked(True)
        viz_grid.addWidget(self.color_chk, 0, 0, 1, 2)
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.setEnabled(False)
        viz_grid.addWidget(QtWidgets.QLabel('Color feature:'), 1, 0)
        viz_grid.addWidget(self.color_combo, 1, 1, 1, 2)
        self.p99_chk = QtWidgets.QCheckBox('vmax=p99')
        self.p99_chk.setChecked(True)
        self.p99_chk.setEnabled(False)
        viz_grid.addWidget(self.p99_chk, 0, 2)

        viz_grid.addWidget(QtWidgets.QLabel('Feature X:'), 2, 0)
        self.x_feature_combo = QtWidgets.QComboBox()
        self.x_feature_combo.setEnabled(False)
        viz_grid.addWidget(self.x_feature_combo, 2, 1, 1, 2)

        viz_grid.addWidget(QtWidgets.QLabel('Feature Y:'), 3, 0)
        self.y_feature_combo = QtWidgets.QComboBox()
        self.y_feature_combo.setEnabled(False)
        viz_grid.addWidget(self.y_feature_combo, 3, 1, 1, 2)

        self.x_log_chk = QtWidgets.QCheckBox('Log X')
        self.x_log_chk.setChecked(False)
        viz_grid.addWidget(self.x_log_chk, 4, 0)

        self.y_log_chk = QtWidgets.QCheckBox('Log Y')
        self.y_log_chk.setChecked(False)
        viz_grid.addWidget(self.y_log_chk, 4, 1)

        self.use_scaled_chk = QtWidgets.QCheckBox('Use scaled features')
        self.use_scaled_chk.setChecked(True)
        self.use_scaled_chk.setToolTip('Use z-score scaled features for visualization (uncheck for raw values)')
        viz_grid.addWidget(self.use_scaled_chk, 4, 2)

        viz_grid.addWidget(QtWidgets.QLabel('DAPI min:'), 5, 0)
        self.dapi_min_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dapi_min_slider.setMinimum(0)
        self.dapi_min_slider.setMaximum(200)
        self.dapi_min_slider.setValue(0)
        self.dapi_min_slider.setTickInterval(10)
        viz_grid.addWidget(self.dapi_min_slider, 5, 1)
        self.dapi_min_label = QtWidgets.QLabel('0.00x')
        viz_grid.addWidget(self.dapi_min_label, 5, 2)

        viz_grid.addWidget(QtWidgets.QLabel('DAPI max:'), 6, 0)
        self.dapi_max_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dapi_max_slider.setMinimum(10)
        self.dapi_max_slider.setMaximum(500)
        self.dapi_max_slider.setValue(100)
        self.dapi_max_slider.setTickInterval(10)
        viz_grid.addWidget(self.dapi_max_slider, 6, 1)
        self.dapi_max_label = QtWidgets.QLabel('1.00x')
        viz_grid.addWidget(self.dapi_max_label, 6, 2)
        self._update_dapi_window_labels()

        # wire color controls
        self.color_chk.stateChanged.connect(self._toggle_color_controls)
        self.color_combo.currentIndexChanged.connect(lambda _: self.update_umap_colors())
        self.p99_chk.stateChanged.connect(lambda _: self.update_umap_colors())
        self.x_feature_combo.currentIndexChanged.connect(lambda _: self.update_feature_scatter())
        self.y_feature_combo.currentIndexChanged.connect(lambda _: self.update_feature_scatter())
        self.x_log_chk.stateChanged.connect(lambda _: self.update_feature_scatter())
        self.y_log_chk.stateChanged.connect(lambda _: self.update_feature_scatter())
        self.use_scaled_chk.stateChanged.connect(lambda _: self.update_umap_colors())
        self.use_scaled_chk.stateChanged.connect(lambda _: self.update_feature_scatter())
        self.dapi_min_slider.valueChanged.connect(self._on_dapi_window_changed)
        self.dapi_max_slider.valueChanged.connect(self._on_dapi_window_changed)
        # Early internal state needed by startup callbacks
        self.display_df = None
        self._toggle_color_controls(self.color_chk.checkState().value)

        # Image panel: 1 column x 3 rows
        image_group = QtWidgets.QGroupBox('Selected Cell Images')
        image_layout = QtWidgets.QHBoxLayout(image_group)
        left_layout.addWidget(image_group, stretch=1)
        self.img_fig = Figure(figsize=(12, 5.8))
        self.img_canvas = FigureCanvas(self.img_fig)
        image_layout.addWidget(self.img_canvas, stretch=1)
        self.img_axes = [self.img_fig.add_subplot(1, 3, i + 1) for i in range(3)]
        for ax in self.img_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.axis('off')

        scalebar_panel = QtWidgets.QWidget()
        scalebar_panel.setFixedWidth(140)
        scalebar_layout = QtWidgets.QVBoxLayout(scalebar_panel)
        scalebar_layout.setContentsMargins(0, 0, 0, 0)
        scalebar_layout.setSpacing(6)
        self.show_scalebar_chk = QtWidgets.QCheckBox('Scale bar\n(10 um)')
        self.show_scalebar_chk.setChecked(False)
        self.show_scalebar_chk.setEnabled(False)
        scalebar_layout.addWidget(self.show_scalebar_chk)
        scalebar_layout.addStretch(1)
        image_layout.addWidget(scalebar_panel)

        self.show_scalebar_chk.stateChanged.connect(self._refresh_selected_images)

        save_group = QtWidgets.QGroupBox('Save')
        save_layout = QtWidgets.QVBoxLayout(save_group)
        save_layout.setContentsMargins(8, 8, 8, 8)
        save_layout.setSpacing(6)
        self.save_img_btn = QtWidgets.QPushButton('Save Cell Images')
        self.save_img_btn.clicked.connect(self.save_current_cell_images)
        self.save_img_btn.setEnabled(False)
        self.save_adata_btn = QtWidgets.QPushButton('Save AnnData (.h5ad)')
        self.save_adata_btn.clicked.connect(self.save_adata)
        self.save_adata_btn.setEnabled(False)
        save_layout.addWidget(self.save_img_btn)
        save_layout.addWidget(self.save_adata_btn)
        left_layout.addWidget(save_group, stretch=0)

        # Plot panel
        plots_group = QtWidgets.QGroupBox('UMAP and Feature Scatter')
        plots_layout = QtWidgets.QVBoxLayout(plots_group)
        right_layout.addWidget(plots_group, stretch=1)
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        plots_layout.addWidget(self.canvas)
        self.plot_grid = self.fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.10], wspace=0.18)
        self.ax_umap = self.fig.add_subplot(self.plot_grid[0, 0])
        self.ax_feat = self.fig.add_subplot(self.plot_grid[0, 1])
        self.colorbar_ax = self.fig.add_subplot(self.plot_grid[0, 2])
        self.colorbar_ax.set_axis_off()
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
        self.feat_coords_nn = None
        self.feat_mean = None
        self.feat_std = None
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
        self.image_pixel_size_um = None
        self.scalebar_artists = []
        # Image cache for lazy-loading (path -> ndarray)
        self.image_cache = {}
        self.image_cache_size_limit = 50  # max images to keep in memory
        self._show_empty_images('')
        self._redraw_plots()

    def _apply_startup_config(self):
        if not self.startup_config:
            return
        if self.startup_config.get('features_path'):
            self.features_path.setText(self.startup_config['features_path'])
        if self.startup_config.get('images_path'):
            self.images_path.setText(self.startup_config['images_path'])
        if self.startup_config.get('channel_map_path'):
            self.channel_map_path.setText(self.startup_config['channel_map_path'])
        if self.startup_config.get('output_path'):
            self.output_path.setText(self.startup_config['output_path'])
        if self.startup_config.get('name_prefix') is not None:
            self.name_prefix.setText(self.startup_config['name_prefix'])

    def _feature_prefix(self) -> str:
        raw = self.name_prefix.text().strip()
        if not raw:
            return ''
        safe = re.sub(r'\s+', '_', raw)
        return f'{safe}_'

    def load_data(self):
        self.log('[LOADING DATA] Starting')
        feats = Path(self.features_path.text())
        imgs = Path(self.images_path.text())
        cmap = Path(self.channel_map_path.text())
        channel_map = load_channel_map(cmap)
        # ensure channel_map is available as an attribute before any method
        # that may reference it (e.g. _rename_feature_columns)
        self.channel_map = channel_map
        df = load_features_table(feats)
        df = index_images(imgs, df, channel_map)
        df = self._rename_feature_columns(df)
        df = self._filter_features_by_available_channels(df)
        df = self._add_area_ratios(df)
        df = self._add_length_width_ratios(df)
        # filter to only rows that have at least one channel path
        path_cols = [str(ch) + '_path' for ch in channel_map.values()]
        available_mask = df[path_cols].notna().any(axis=1)
        df_filtered = df[available_mask].reset_index(drop=True)
        # count available image paths and rows
        found_counts = int(df_filtered[path_cols].notna().sum().sum())
        rows_with_images = int(df_filtered.shape[0])
        self.df = df_filtered
        self.display_df = self.df.reset_index(drop=True)
        self.log(f'[LOADED] {len(df)} objects; {rows_with_images} objects have at least one image; total channel paths found: {found_counts}')
        self._populate_feature_controls()
        self._update_feature_coords()
        self._redraw_plots()
        self.canvas.draw()
        QtCore.QCoreApplication.processEvents()
        self._estimate_dapi_scale()

    def _rename_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        used = set(df.columns)
        channel_aliases = {}
        for raw_key, friendly in self.channel_map.items():
            key = raw_key.lower()
            channel_aliases[key] = friendly
            if key.startswith('ch'):
                digits = key[2:]
                if digits.isdigit():
                    channel_aliases[f'm{int(digits):02d}'] = friendly
                    channel_aliases[f'm{int(digits)}'] = friendly
        channel_aliases.setdefault('m01', 'BF')
        channel_aliases.setdefault('m1', 'BF')
        channel_aliases.setdefault('m07', 'DAPI')
        channel_aliases.setdefault('m7', 'DAPI')
        channel_aliases.setdefault('ch1', 'BF')
        channel_aliases.setdefault('ch7', 'DAPI')
        for col in list(df.columns):
            if col == 'Object Number' or col.endswith('_path'):
                continue
            new_col = col
            for alias, friendly in channel_aliases.items():
                pattern = re.compile(rf'(^|[^0-9A-Za-z]){re.escape(alias)}(?=$|[^0-9A-Za-z])', re.IGNORECASE)

                def _sub(match):
                    return f'{match.group(1)}{friendly}'

                new_col = pattern.sub(_sub, new_col)
            new_col = re.sub(r'\s+', ' ', new_col).strip()
            for friendly in sorted(set(channel_aliases.values()) | {'BF', 'DAPI'}, key=len, reverse=True):
                dup_pattern = re.compile(
                    rf'(?i)(?:(?<=^)|(?<=[_\s-])){re.escape(friendly)}(?:[_\s-]+{re.escape(friendly)})+'
                )
                new_col = dup_pattern.sub(friendly, new_col)
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

    def _add_area_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        area_cols = [c for c in df.columns if c != 'Object Number' and not c.endswith('_path') and 'area' in c.lower()]
        if len(area_cols) < 2:
            return df
        new_cols = {}
        for i, numerator in enumerate(area_cols):
            numerator_vals = pd.to_numeric(df[numerator], errors='coerce').astype(float)
            for denominator in area_cols[i + 1:]:
                denominator_vals = pd.to_numeric(df[denominator], errors='coerce').astype(float)
                ratio_name = f'{numerator}_to_{denominator}'
                ratio_vals = np.full(len(df), np.nan, dtype=np.float32)
                valid = denominator_vals.to_numpy(dtype=float) != 0
                ratio_vals[valid] = (numerator_vals.to_numpy(dtype=float)[valid] / denominator_vals.to_numpy(dtype=float)[valid]).astype(np.float32)
                new_cols[ratio_name] = ratio_vals
                reverse_name = f'{denominator}_to_{numerator}'
                reverse_vals = np.full(len(df), np.nan, dtype=np.float32)
                valid_rev = numerator_vals.to_numpy(dtype=float) != 0
                reverse_vals[valid_rev] = (denominator_vals.to_numpy(dtype=float)[valid_rev] / numerator_vals.to_numpy(dtype=float)[valid_rev]).astype(np.float32)
                new_cols[reverse_name] = reverse_vals
        for col_name, values in new_cols.items():
            if col_name not in df.columns:
                df[col_name] = values
        if new_cols:
            self.log(f'[LOAD] Added {len(new_cols)} area-ratio features')
        return df

    def _filter_features_by_available_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        allowed_channel_numbers = set()
        allowed_channel_labels = set()
        for key in self.channel_map.keys():
            low = str(key).strip().lower()
            m = re.match(r'^(?:ch|m)0*(\d{1,2})$', low)
            if m:
                allowed_channel_numbers.add(int(m.group(1)))
        for val in self.channel_map.values():
            low_val = str(val).strip().lower()
            if low_val:
                allowed_channel_labels.add(low_val)
        # Keep legacy defaults for BF/DAPI when channel numbers are implied.
        if 'ch1' in {str(k).strip().lower() for k in self.channel_map.keys()}:
            allowed_channel_numbers.add(1)
        if 'ch7' in {str(k).strip().lower() for k in self.channel_map.keys()}:
            allowed_channel_numbers.add(7)

        if not allowed_channel_numbers:
            return df

        keep_cols = []
        dropped = []
        for col in df.columns:
            if col == 'Object Number' or col.endswith('_path'):
                keep_cols.append(col)
                continue

            # Columns like Intensity_MC_DAPI / Max Pixel_MC_BF encode channel name, not channel number.
            mc_match = re.search(r'(?i)_MC_([^_]+)$', col)
            if mc_match:
                mc_label = mc_match.group(1).strip().lower()
                if mc_label in allowed_channel_labels:
                    keep_cols.append(col)
                else:
                    dropped.append(col)
                continue

            numeric_hits = [int(n) for n in re.findall(r'(?i)(?:^|[^0-9A-Za-z])(?:m|ch)0*(\d{1,2})(?=$|[^0-9A-Za-z])', col)]
            if not numeric_hits:
                # Channel-agnostic features are retained.
                keep_cols.append(col)
                continue

            if any(n in allowed_channel_numbers for n in numeric_hits):
                keep_cols.append(col)
            else:
                dropped.append(col)

        if dropped:
            self.log(f'[LOAD] Filtered out {len(dropped)} feature columns not present in channel map')
        return df[keep_cols]

    def _add_length_width_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c != 'Object Number' and not c.endswith('_path')]
        pairs = {}
        for col in feature_cols:
            low = col.lower()
            if 'length' not in low and 'width' not in low:
                continue
            parts = [p for p in re.split(r'[_\-\s]+', col) if p]
            role = None
            base_parts = []
            for p in parts:
                pl = p.lower()
                if pl == 'length':
                    role = 'length'
                    continue
                if pl == 'width':
                    role = 'width'
                    continue
                base_parts.append(pl)
            if role is None or not base_parts:
                continue
            base = ' '.join(base_parts)
            slot = pairs.setdefault(base, {})
            slot[role] = col

        added = 0
        for base, cols in pairs.items():
            length_col = cols.get('length')
            width_col = cols.get('width')
            if not length_col or not width_col:
                continue
            ratio_name = f'{length_col}_to_{width_col}'
            if ratio_name in df.columns:
                continue
            length_vals = pd.to_numeric(df[length_col], errors='coerce').astype(float)
            width_vals = pd.to_numeric(df[width_col], errors='coerce').astype(float)
            ratio_vals = np.full(len(df), np.nan, dtype=np.float32)
            valid = width_vals.to_numpy(dtype=float) != 0
            ratio_vals[valid] = (length_vals.to_numpy(dtype=float)[valid] / width_vals.to_numpy(dtype=float)[valid]).astype(np.float32)
            df[ratio_name] = ratio_vals
            added += 1

        if added:
            self.log(f'[LOAD] Added {added} length/width ratio features')
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

        enabled = (self.display_df is not None) and self.color_chk.isChecked()
        self.color_combo.setEnabled(enabled)
        self.p99_chk.setEnabled(enabled)
        self.x_feature_combo.setEnabled(True)
        self.y_feature_combo.setEnabled(True)

        # default feature scatter: Area BF vs DAPI intensity when available
        x_default = self._find_feature_name(['Area', 'BF'])
        y_default = self._find_feature_name(['DAPI', 'Intensity'])
        if self.startup_x_feature:
            x_default = self.startup_x_feature if self.startup_x_feature in feature_cols else x_default
        if self.startup_y_feature:
            y_default = self.startup_y_feature if self.startup_y_feature in feature_cols else y_default
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
            self.dapi_scale = max(float(np.mean(vals)), 1.0)
            self.dapi_min_slider.setValue(0)
            self.dapi_max_slider.setValue(100)
            self._update_dapi_window_labels()
            self.log(f'[LOAD] DAPI fixed scale set to {self.dapi_scale:.2f} (p99.5 mean from {len(vals)} sampled cells)')

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

    def _update_dapi_window_labels(self):
        min_scale = self.dapi_min_slider.value() / 100.0
        max_scale = self.dapi_max_slider.value() / 100.0
        self.dapi_min_label.setText(f'{min_scale:.2f}x')
        self.dapi_max_label.setText(f'{max_scale:.2f}x')

    def _on_dapi_window_changed(self, _value):
        self._update_dapi_window_labels()
        if self.current_index is not None:
            self._show_selected_images(self.current_index)

    def _refresh_selected_images(self, *_args):
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
        self.log(f'[PERFORMING UMAP] Computing with {n_samples} samples and {n_features} features')
        QtCore.QCoreApplication.processEvents()
        try:
            adata = ad.AnnData(X)
            # Store feature names in var so they persist through save/load
            adata.var.index = features
            adata.obs = pd.DataFrame(index=range(adata.n_obs))
            if 'Object Number' in self.df.columns:
                adata.obs['Object Number'] = self.df['Object Number'].values
            for col in self.df.columns:
                if col.endswith('_path'):
                    adata.obs[col] = self.df[col].values
            self.display_df = self.df.reset_index(drop=True)
            
            # Store raw features, then scale for PCA
            adata.layers['raw'] = adata.X.copy()
            sc.pp.scale(adata)
            adata.layers['scaled'] = adata.X.copy()
            
            # choose n_comps safely based on adata
            n_samples = int(adata.n_obs)
            n_features = int(adata.n_vars)
            min_dim = min(n_samples, n_features)
            if min_dim <= 1:
                msg = f'UMAP skipped: need at least 2 samples and 2 features (found samples={n_samples}, features={n_features})'
                self.log(f'[UMAP] {msg}')
                QtWidgets.QMessageBox.warning(self, 'UMAP skipped', msg)
                return
            # PCA via scanpy/ sklearn with svd_solver='arpack' requires n_comps < min(n_samples, n_features)
            # choose at most 50 components but strictly less than min_dim
            n_comps = max(1, min(50, min_dim - 1))
            self.log(f'[UMAP] using n_comps={n_comps} for PCA')
            sc.pp.pca(adata, n_comps=n_comps)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            self.adata = adata
            self.coords = adata.obsm['X_umap']
            self.nn_umap = NearestNeighbors(n_neighbors=1).fit(self.coords)

            # ensure color controls reflect current data/state
            try:
                self._toggle_color_controls(self.color_chk.checkState().value)
            except Exception:
                pass

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
            self.feat_coords_nn = None
            self.feat_mean = None
            self.feat_std = None
            self.nn_feat = None
            return
        
        # Use scaled or raw values based on checkbox
        if self.use_scaled_chk.isChecked() and self.adata is not None and 'scaled' in self.adata.layers:
            x_idx = list(self.adata.var.index).index(x_name) if x_name in self.adata.var.index else None
            y_idx = list(self.adata.var.index).index(y_name) if y_name in self.adata.var.index else None
            if x_idx is not None and y_idx is not None:
                xvals = self.adata.layers['scaled'][:, x_idx].astype(np.float32)
                yvals = self.adata.layers['scaled'][:, y_idx].astype(np.float32)
            else:
                xvals = pd.to_numeric(self.display_df[x_name], errors='coerce').fillna(0).values.astype(np.float32)
                yvals = pd.to_numeric(self.display_df[y_name], errors='coerce').fillna(0).values.astype(np.float32)
        else:
            # Use raw values from display_df
            xvals = pd.to_numeric(self.display_df[x_name], errors='coerce').fillna(0).values.astype(np.float32)
            yvals = pd.to_numeric(self.display_df[y_name], errors='coerce').fillna(0).values.astype(np.float32)
        
        if self.x_log_chk.isChecked():
            xvals = np.log1p(np.clip(xvals, 0, None))
        if self.y_log_chk.isChecked():
            yvals = np.log1p(np.clip(yvals, 0, None))
        self.feat_coords = np.column_stack([xvals, yvals])
        self.feat_mean = np.array([float(np.mean(xvals)), float(np.mean(yvals))], dtype=np.float32)
        self.feat_std = np.array([float(np.std(xvals)), float(np.std(yvals))], dtype=np.float32)
        self.feat_std = np.where(self.feat_std <= 1e-9, 1.0, self.feat_std)
        self.feat_coords_nn = (self.feat_coords - self.feat_mean) / self.feat_std
        self.nn_feat = NearestNeighbors(n_neighbors=1).fit(self.feat_coords_nn)

    def _draw_umap_scatter(self):
        if self.coords is None:
            return None

        vals, feat_name, vmax = self._get_color_feature_values()
        if vals is not None:
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
            self._draw_colorbar(sca, feat_name)
            return sca

        return self.ax_umap.scatter(self.coords[:, 0], self.coords[:, 1], s=16, c='C0', picker=True)

    def _draw_feature_scatter(self):
        if self.feat_coords is None:
            return None
        x_name = self.x_feature_combo.currentText().strip()
        y_name = self.y_feature_combo.currentText().strip()
        vals, _, vmax = self._get_color_feature_values()
        if vals is not None:
            sca = self.ax_feat.scatter(
                self.feat_coords[:, 0],
                self.feat_coords[:, 1],
                c=vals,
                s=14,
                cmap='viridis',
                vmin=0,
                vmax=vmax,
                alpha=0.9,
                picker=True,
            )
        else:
            sca = self.ax_feat.scatter(self.feat_coords[:, 0], self.feat_coords[:, 1], s=14, c='0.35', alpha=0.9, picker=True)
        x_label = f'log1p({x_name})' if self.x_log_chk.isChecked() else x_name
        y_label = f'log1p({y_name})' if self.y_log_chk.isChecked() else y_name
        self.ax_feat.set_xlabel(x_label)
        self.ax_feat.set_ylabel(y_label)
        return sca

    def _get_color_feature_values(self):
        if not self.color_chk.isChecked() or self.display_df is None:
            return None, None, None
        feat_name = self.color_combo.currentText().strip()
        if not feat_name or feat_name not in self.display_df.columns:
            return None, None, None
        
        # Use scaled or raw values based on checkbox
        if self.use_scaled_chk.isChecked() and self.adata is not None and 'scaled' in self.adata.layers:
            # Get scaled values from adata
            feat_idx = list(self.adata.var.index).index(feat_name) if feat_name in self.adata.var.index else None
            if feat_idx is not None:
                vals = self.adata.layers['scaled'][:, feat_idx]
            else:
                vals = pd.to_numeric(self.display_df[feat_name], errors='coerce').fillna(0).values
        else:
            # Use raw values from display_df
            vals = pd.to_numeric(self.display_df[feat_name], errors='coerce').fillna(0).values
        
        vmax = None
        if self.p99_chk.isChecked():
            try:
                vmax = float(np.nanpercentile(vals, 99))
            except Exception:
                vmax = None
        return vals, feat_name, vmax

    def _draw_colorbar(self, mappable, feat_name):
        self.colorbar = None
        if self.colorbar_ax is None or self.colorbar_ax.figure is None:
            self.colorbar_ax = self.fig.add_subplot(self.plot_grid[0, 2])
        self.colorbar_ax.clear()
        self.colorbar_ax.set_axis_off()
        if feat_name:
            try:
                self.colorbar_ax.set_axis_on()
                self.colorbar = self.fig.colorbar(mappable, cax=self.colorbar_ax)
                self.colorbar.set_label(feat_name)
            except Exception:
                self.colorbar = None

    def _redraw_plots(self):
        self.ax_umap.clear()
        self.ax_feat.clear()
        if self.coords is None:
            self.ax_umap.text(0.5, 0.5, 'UMAP not computed yet', ha='center', va='center', transform=self.ax_umap.transAxes)
            self.ax_umap.set_xticks([])
            self.ax_umap.set_yticks([])
            self.ax_umap.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.umap_scatter = None
        else:
            self.umap_scatter = self._draw_umap_scatter()
        if self.display_df is not None:
            self.feat_scatter = self._draw_feature_scatter()
        else:
            self.ax_feat.set_xticks([])
            self.ax_feat.set_yticks([])
            self.ax_feat.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.feat_scatter = None
        self.ax_umap.set_box_aspect(1)
        self.ax_feat.set_box_aspect(1)
        self.fig.tight_layout(pad=0.8, rect=[0.01, 0.02, 0.995, 0.98])

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
        if self.display_df is None:
            return
        self._redraw_plots()

    def update_feature_scatter(self):
        if self.display_df is None:
            return
        self._update_feature_coords()
        self._redraw_plots()

    def on_click(self, event):
        if event.inaxes not in (self.ax_umap, self.ax_feat):
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if event.inaxes == self.ax_umap and self.coords is not None and self.nn_umap is not None:
            self._maybe_select_nearest(self.coords, self.nn_umap, x, y, self.ax_umap)
        elif event.inaxes == self.ax_feat and self.feat_coords_nn is not None and self.nn_feat is not None:
            self._maybe_select_nearest(self.feat_coords_nn, self.nn_feat, x, y, self.ax_feat, transform='feature')

    def _maybe_select_nearest(self, coords, nn_model, x, y, axis, transform=None):
        if coords is None or nn_model is None:
            return
        if transform == 'feature' and self.feat_mean is not None and self.feat_std is not None:
            x = (float(x) - float(self.feat_mean[0])) / float(self.feat_std[0])
            y = (float(y) - float(self.feat_mean[1])) / float(self.feat_std[1])
        dist, idx = nn_model.kneighbors([[x, y]], return_distance=True)
        nearest_dist = float(dist[0][0])
        if transform == 'feature':
            threshold = 0.45
        else:
            x_min, x_max = axis.get_xlim()
            y_min, y_max = axis.get_ylim()
            span = max(abs(x_max - x_min), abs(y_max - y_min), 1e-9)
            threshold = span * 0.08
        if nearest_dist > threshold:
            return
        self._select_index(int(idx[0][0]))

    def _select_index(self, i: int):
        self.current_index = int(i)
        self._draw_selection_markers()
        self.canvas.draw()
        self._show_selected_images(self.current_index)
        self.save_img_btn.setEnabled(True)

    def _show_selected_images(self, i: int):
        if self.display_df is None or i < 0 or i >= len(self.display_df):
            self._show_empty_images('')
            return
        obj_row = self.display_df.iloc[i]
        obj_id = str(obj_row.get('Object Number'))

        bf_path = obj_row.get('BF_path')
        dapi_path = obj_row.get('DAPI_path')
        try:
            bf = self._load_image_cached(Path(bf_path)) if bf_path else np.zeros((64, 64), dtype=np.float32)
        except Exception as e:
            self.log(f'[ERROR] Failed to read BF image for {obj_row.get("Object Number")}: {e}')
            bf = np.zeros((64, 64), dtype=np.float32)
        try:
            dapi = self._load_image_cached(Path(dapi_path)) if dapi_path else np.zeros_like(bf)
        except Exception as e:
            self.log(f'[ERROR] Failed to read DAPI image for {obj_row.get("Object Number")}: {e}')
            dapi = np.zeros_like(bf)

        pixel_size_um = None
        for candidate in (bf_path, dapi_path):
            if candidate:
                pixel_size_um = get_pixel_size_um(Path(candidate))
                if pixel_size_um is not None:
                    break
        if pixel_size_um is not None:
            self.image_pixel_size_um = float(pixel_size_um)
            self.show_scalebar_chk.setEnabled(True)
        else:
            self.image_pixel_size_um = None
            self.show_scalebar_chk.setChecked(False)
            self.show_scalebar_chk.setEnabled(False)

        # BF is always shown fully as the base channel.
        bf_disp = normalize_image(bf)

        # DAPI uses background subtraction and fixed normalization for cross-cell comparison.
        dapi_bg = subtract_background(dapi)
        dapi_min = (self.dapi_min_slider.value() / 100.0) * self.dapi_scale if self.dapi_scale is not None else 0.0
        dapi_max = (self.dapi_max_slider.value() / 100.0) * self.dapi_scale if self.dapi_scale is not None else max(float(np.nanpercentile(dapi_bg, 99.5)), 1.0)
        if dapi_max <= dapi_min:
            dapi_max = dapi_min + 1.0
        dapi_disp = np.clip((dapi_bg - dapi_min) / (dapi_max - dapi_min), 0.0, 1.0)

        overlay = overlay_images(bf, dapi, alpha=0.85, dapi_min=dapi_min, dapi_max=dapi_max)

        self.img_axes[0].clear()
        self.img_axes[0].imshow(bf_disp, cmap='gray')
        self.img_axes[0].set_title('BF', color='#444444', fontweight='bold')
        self.img_axes[1].clear()
        self.img_axes[1].imshow(dapi_disp, cmap='gray')
        self.img_axes[1].set_title('DAPI', color='#00bcd4', fontweight='bold')
        self.img_axes[2].clear()
        self.img_axes[2].imshow(np.array(overlay))
        self.img_axes[2].set_title('Overlay', color='#7b1fa2', fontweight='bold')

        self.img_fig.suptitle(f'Cell {obj_id}', fontsize=12, y=0.985)

        self._clear_scalebars()
        if self.show_scalebar_chk.isChecked():
            self._draw_scalebar(self.img_axes, bf_disp.shape)
        for ax in self.img_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.axis('off')
        self.img_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.80, wspace=0.06)
        self.img_canvas.draw()

    def _show_empty_images(self, message: str):
        self.img_fig.suptitle('')
        for ax in self.img_axes:
            ax.clear()
            if message:
                ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, color='0.4')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.axis('off')
        self.img_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.80, wspace=0.06)
        self.img_canvas.draw()

    def _load_image_cached(self, path: Path) -> np.ndarray:
        """Load image with in-memory cache to avoid repeated disk I/O (lazy-loading optimization)."""
        path_key = str(path)
        if path_key in self.image_cache:
            return self.image_cache[path_key]
        
        # Load from disk
        try:
            img = load_image(path)
        except Exception as e:
            self.log(f'[WARNING] Failed to load image {path}: {e}')
            return np.zeros((64, 64), dtype=np.float32)
        
        # Add to cache; remove oldest if limit exceeded
        self.image_cache[path_key] = img
        if len(self.image_cache) > self.image_cache_size_limit:
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]
        
        return img

    def _clear_scalebars(self):
        for artist in self.scalebar_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.scalebar_artists = []

    def _draw_scalebar(self, axes, image_shape):
        pixel_size_um = self.image_pixel_size_um
        if pixel_size_um is None or pixel_size_um <= 0:
            return
        bar_um = 10.0
        bar_px = max(int(round(bar_um / pixel_size_um)), 1)
        for ax in axes:
            scalebar = AnchoredSizeBar(
                ax.transData,
                bar_px,
                '',
                loc='lower right',
                pad=0.0,
                borderpad=0.0,
                sep=2,
                color='white',
                frameon=False,
                size_vertical=max(int(round(image_shape[0] * 0.01)), 1),
                fontproperties=FontProperties(size=8, weight='bold'),
                bbox_to_anchor=(0.95, 0.01),
                bbox_transform=ax.transAxes,
            )
            ax.add_artist(scalebar)
            self.scalebar_artists.append(scalebar)

    def save_current_cell_images(self):
        if self.current_index is None:
            QtWidgets.QMessageBox.warning(self, 'No selection', 'No cell selected')
            return
        out_dir = Path(self.output_path.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._feature_prefix()
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
        dapi_min = (self.dapi_min_slider.value() / 100.0) * self.dapi_scale if self.dapi_scale is not None else 0.0
        dapi_max = (self.dapi_max_slider.value() / 100.0) * self.dapi_scale if self.dapi_scale is not None else max(float(np.nanpercentile(dapi_bg, 99.5)), 1.0)
        if dapi_max <= dapi_min:
            dapi_max = dapi_min + 1.0
        dapi_n = (np.clip((dapi_bg - dapi_min) / (dapi_max - dapi_min), 0.0, 1.0) * 255).astype(np.uint8)
        from PIL import Image
        bf_img = Image.fromarray(bf_n)
        dapi_img = Image.fromarray(dapi_n)
        overlay_img = overlay_images(bf, dapi, alpha=0.85, dapi_min=dapi_min, dapi_max=dapi_max)

        # save PNG and TIFF
        bf_png = out_dir / f'{prefix}{obj}_BF.png'
        dapi_png = out_dir / f'{prefix}{obj}_DAPI.png'
        overlay_png = out_dir / f'{prefix}{obj}_overlay.png'

        bf_img.save(str(bf_png))
        dapi_img.save(str(dapi_png))
        overlay_img.save(str(overlay_png))

        self.log(f'[EXPORT] Saved images for {obj} to {out_dir}')

    def save_adata(self):
        if self.adata is None:
            QtWidgets.QMessageBox.warning(self, 'No data', 'No AnnData available')
            return
        out_dir = Path(self.output_path.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._feature_prefix()
        out_file = out_dir / f'{prefix}adata.h5ad'
        try:
            # Store display settings in uns for session restoration
            self.adata.uns['session_settings'] = {
                'dapi_scale': float(self.dapi_scale) if self.dapi_scale else None,
                'color_feature': self.color_combo.currentText().strip(),
                'use_p99': self.p99_chk.isChecked(),
                'x_feature': self.x_feature_combo.currentText().strip(),
                'y_feature': self.y_feature_combo.currentText().strip(),
                'x_log': self.x_log_chk.isChecked(),
                'y_log': self.y_log_chk.isChecked(),
                'color_enabled': self.color_chk.isChecked(),
                'use_scaled': self.use_scaled_chk.isChecked(),
            }
            self.adata.write_h5ad(str(out_file))
            self.log(f'[EXPORT] Saved AnnData to {out_file}')
        except Exception as e:
            self.log(f'[ERROR] Saving AnnData failed: {e}')
            QtWidgets.QMessageBox.critical(self, 'Save error', str(e))

    def load_adata_file(self):
        """Load a previously saved h5ad file with UMAP and display settings."""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select AnnData file', str(Path('.')), 'H5AD files (*.h5ad);;All files (*)'
        )
        if not fn:
            return
        try:
            self.log(f'[LOADING H5AD] {fn}')
            self.adata = ad.read_h5ad(fn)
            
            # Restore feature display_df with proper column names and obs columns
            feature_cols = list(self.adata.var.index) if hasattr(self.adata, 'var') else []
            self.display_df = pd.DataFrame(
                self.adata.X,
                columns=feature_cols
            )
            # Add all obs columns back
            for col in self.adata.obs.columns:
                self.display_df[col] = self.adata.obs[col].values
            self.log(f'[H5AD] Restored {len(feature_cols)} features and {len(self.adata.obs.columns)} obs columns')
            
            # Restore UMAP coordinates
            if 'X_umap' in self.adata.obsm:
                self.coords = self.adata.obsm['X_umap']
                self.nn_umap = NearestNeighbors(n_neighbors=1).fit(self.coords)
            else:
                self.coords = None
                self.log('[WARNING] No X_umap found in AnnData')
            
            # Load display settings if available
            if 'session_settings' in self.adata.uns:
                settings = self.adata.uns['session_settings']
                if settings.get('dapi_scale'):
                    self.dapi_scale = float(settings['dapi_scale'])
            
            self.df = self.display_df.copy()
            self._populate_feature_controls()
            
            # Restore color and scatter settings from uns
            if 'session_settings' in self.adata.uns:
                settings = self.adata.uns['session_settings']
                if settings.get('color_enabled'):
                    self.color_chk.setChecked(True)
                    if settings.get('color_feature'):
                        idx = self.color_combo.findText(settings['color_feature'])
                        if idx >= 0:
                            self.color_combo.setCurrentIndex(idx)
                    self.p99_chk.setChecked(settings.get('use_p99', True))
                if settings.get('x_feature'):
                    idx = self.x_feature_combo.findText(settings['x_feature'])
                    if idx >= 0:
                        self.x_feature_combo.setCurrentIndex(idx)
                if settings.get('y_feature'):
                    idx = self.y_feature_combo.findText(settings['y_feature'])
                    if idx >= 0:
                        self.y_feature_combo.setCurrentIndex(idx)
                self.x_log_chk.setChecked(settings.get('x_log', False))
                self.y_log_chk.setChecked(settings.get('y_log', False))
                self.use_scaled_chk.setChecked(settings.get('use_scaled', True))
            
            self._update_feature_coords()
            self._toggle_color_controls(self.color_chk.checkState().value)
            self._redraw_plots()
            self.canvas.draw()
            self.save_adata_btn.setEnabled(True)
            self.save_img_btn.setEnabled(True)
            self.log(f'[H5AD] Loaded successfully; UMAP ready')
        except Exception as e:
            self.log(f'[ERROR] Loading H5AD failed: {e}')
            QtWidgets.QMessageBox.critical(self, 'Load H5AD error', str(e))



def run_app(argv=None):
    args = parse_args(argv)
    startup_config = {
        'features_path': args.features_path,
        'images_path': args.images_path,
        'channel_map_path': args.channel_map_path,
        'output_path': args.output_path,
        'name_prefix': args.name_prefix,
        'x_feature': args.x_feature,
        'y_feature': args.y_feature,
    }
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(startup_config=startup_config)
    win.showMaximized()
    if args.autoload:
        win.load_data()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()

