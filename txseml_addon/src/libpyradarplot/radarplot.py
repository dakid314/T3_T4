'''
Author: George Zhao
Date: 2021-11-06 09:24:32
LastEditors: George Zhao
LastEditTime: 2022-05-21 12:01:48
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib.cm as cm


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def draw_radar(
    path_to_out: str,
    df: np.ndarray,
    df_legend_label: list,
    axisname: list,
    title: str = None,
    axisrgrids: list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
):
    theta = radar_factory(
        len(axisname),
        frame='polygon'
    )

    spoke_labels = list(axisname)

    case_data = df

    colors_list = cm.get_cmap(name='hsv_r', lut=None)(
        np.linspace(start=0, stop=1, num=case_data.shape[0] + 1))

    fig, ax = plt.subplots(
        figsize=(10.8, 5.4),
        subplot_kw=dict(projection='radar')
    )
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids(axisrgrids)
    if title is not None:
        ax.set_title(title, position=(0.5, 1.1), ha='center')

    for d, c in zip(case_data, colors_list):
        line = ax.plot(theta, d, color=c)
        ax.fill(theta, d, color=c, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    legend = ax.legend(
        df_legend_label,
        loc=(1.2, 0.),
        # loc=(0.9, .95),
        labelspacing=0.1,
        fontsize='small'
    )
    plt.tight_layout()
    plt.savefig(path_to_out)
    plt.close(fig)
