"""Figure options and helper functions for plotting."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def cm2inch(x: float, y: float) -> tuple[float, float]:
    """Convert cm to inch."""
    inch = 2.54
    return x / inch, y / inch


def panel_label(
    fig: Figure, ax: Axes, label: str, x: float = 0.0, y: float = 0.0
) -> None:
    """Add panel label to figure."""
    trans = mtransforms.ScaledTranslation(x, y, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="large",
    )


# Formatter for log axis ticks
formatter = mticker.FuncFormatter(lambda y, _: "{:.16g}".format(y))
loc_major = mticker.LogLocator(
    base=10.0,
    subs=(
        0.1,
        1.0,
    ),
    numticks=12,
)
loc_min = mticker.LogLocator(
    base=10.0, subs=tuple(jnp.arange(0.1, 1.0, 0.1)), numticks=12
)

# Colorblind-friendly colors from https://arxiv.org/abs/2107.02270,
# see also https://github.com/matplotlib/matplotlib/issues/9460.
petroff10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

# Color maps
cmap_grays = plt.get_cmap("gray_r")
cmap_blues = plt.get_cmap("Blues")
cmap_oranges = plt.get_cmap("Oranges")
cmap_purples = plt.get_cmap("Purples")
