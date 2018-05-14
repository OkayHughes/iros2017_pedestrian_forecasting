"""Varidis module
Contains the colormap `varidis`, which is viridis but with transparency proportional to the color,
i.e. purple is transparent, yellow-green is fully opaque, with a linear interpolation in between.
"""
from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
_cmap = cm.viridis # pylint: disable=no-member
varidis = _cmap(np.arange(_cmap.N))
varidis[:, -1] = np.linspace(0, 1, _cmap.N)
varidis = ListedColormap(varidis)
