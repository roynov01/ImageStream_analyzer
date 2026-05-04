"""
Visualization utilities for image display and interactive graph analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from typing import Callable, Optional, List, Tuple


class InteractiveImageViewer:
    """
    Interactive image viewer with click detection and ROI selection.
    """
    
    def __init__(self, image: np.ndarray, title: str = "Interactive Image Viewer"):
        """
        Initialize the interactive image viewer.
        
        Parameters
        ----------
        image : np.ndarray
            Image data to display
        title : str
            Title for the plot
        """
        self.image = image
        self.title = title
        self.fig = None
        self.ax = None
        self.clicked_points = []
        
    def show(self) -> None:
        """Display the image with interactive features."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        self.ax.imshow(self.image, cmap='viridis')
        self.ax.set_title(self.title)
        
        # Enable click detection
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        plt.tight_layout()
        plt.show()
    
    def _on_click(self, event) -> None:
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        self.clicked_points.append((x, y))
        
        # Draw marker at clicked point
        self.ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
        self.fig.canvas.draw()
        
        print(f"Clicked at: ({x}, {y})")


def display_image(image: np.ndarray, title: str = "Image", cmap: str = "viridis") -> None:
    """
    Display a single image.
    
    Parameters
    ----------
    image : np.ndarray
        Image data
    title : str
        Title for the plot
    cmap : str
        Colormap to use
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
