from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
_cmap = cm.viridis
varidis = _cmap(np.arange(_cmap.N))
varidis[:,-1] = np.linspace(0, 1, _cmap.N)
varidis = ListedColormap(varidis)
