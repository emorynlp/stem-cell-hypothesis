# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:52
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


def plot_bar(data: Dict[str, List], legends=None, title=None, ylabel=None, xlabel=None):
    N = len(data)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27  # the width of the bars

    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    values = list(data.values())
    for i in range(len(values[0])):
        rects = ax.bar(ind + width * i, [x[i] for x in values], width)
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * h, '%.1f' % h, ha='center', va='bottom')
    if legends:
        ax.legend(legends)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple(data.keys()))
    fig.suptitle(title)
    return fig