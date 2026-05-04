"""PyQt6 GUI application for ImageStream analysis (minimal working version)."""
import sys
from pathlib import Path
from typing import List

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import load_channel_map, load_features_table, index_images
from .image_io import load_image, overlay_images

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

        self.features_path = QtWidgets.QLineEdit('data/6_65_allstains_1.txt')
        self.images_path = QtWidgets.QLineEdit('data/65_imaages/All')
        self.channel_map_path = QtWidgets.QLineEdit('data/channel_map.json')
        load_btn = QtWidgets.QPushButton('Load Data')
        load_btn.clicked.connect(self.load_data)

        controls.addWidget(QtWidgets.QLabel('Features:'))
        controls.addWidget(self.features_path)
        controls.addWidget(QtWidgets.QLabel('Images:'))
        controls.addWidget(self.images_path)
        controls.addWidget(QtWidgets.QLabel('Channel map:'))
        controls.addWidget(self.channel_map_path)
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

        # Matplotlib canvas for plot
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Image display area
        self.img_fig = Figure(figsize=(6, 3))
        self.img_canvas = FigureCanvas(self.img_fig)
        layout.addWidget(self.img_canvas)
        self.img_axes = [self.img_fig.add_subplot(1, 3, i + 1) for i in range(3)]

        # Internal
        self.df = None
        self.adata = None
        self.coords = None
        self.nn = None

    def load_data(self):
        feats = Path(self.features_path.text())
        imgs = Path(self.images_path.text())
        cmap = Path(self.channel_map_path.text())
        channel_map = load_channel_map(cmap)
        df = load_features_table(feats)
        df = index_images(imgs, df, channel_map)
        self.df = df
        self.channel_map = channel_map
        self.feature_list.clear()
        for c in df.columns:
            if c.endswith('_path'):
                continue
            if c == 'Object Number':
                continue
            self.feature_list.addItem(c)
        QtWidgets.QMessageBox.information(self, 'Loaded', f'Loaded {len(df)} objects')

    def compute_umap(self):
        if self.df is None:
            return
        items = self.feature_list.selectedItems()
        if not items:
            features = [c for c in self.df.columns if c.endswith('_path') is False and c != 'Object Number']
        else:
            features = [it.text() for it in items]
        X = self.df[features].fillna(0).values
        adata = ad.AnnData(X)
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        self.adata = adata
        self.coords = adata.obsm['X_umap']
        self.ax.clear()
        self.ax.scatter(self.coords[:, 0], self.coords[:, 1], s=6)
        self.ax.set_title('UMAP')
        self.canvas.draw()
        # build nearest neighbor index
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.coords)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if self.coords is None:
            return
        dist, idx = self.nn.kneighbors([[x, y]], return_distance=True)
        i = int(idx[0][0])
        obj_row = self.df.iloc[i]
        # load images
        bf_path = obj_row.get('BF_path')
        dapi_path = obj_row.get('DAPI_path')
        if bf_path:
            bf = load_image(Path(bf_path))
        else:
            bf = np.zeros((64, 64))
        if dapi_path:
            dapi = load_image(Path(dapi_path))
        else:
            dapi = np.zeros_like(bf)

        overlay = overlay_images(bf, dapi)

        # show images
        self.img_axes[0].clear()
        self.img_axes[0].imshow(bf, cmap='gray')
        self.img_axes[0].set_title('BF')
        self.img_axes[1].clear()
        self.img_axes[1].imshow(dapi, cmap='gray')
        self.img_axes[1].set_title('DAPI')
        self.img_axes[2].clear()
        self.img_axes[2].imshow(overlay)
        self.img_axes[2].set_title('Overlay')
        for ax in self.img_axes:
            ax.axis('off')
        self.img_canvas.draw()


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()
