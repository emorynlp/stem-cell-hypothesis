# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-08 17:51
import os
from typing import List
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from elit.components.mtl.gated.draw_attn import heatmap
from elit.components.mtl.gated.gated_mtl import GatedMultiTaskLearning
from stem_cell_hypothesis import cdroot

cdroot()


def draw(gates: torch.Tensor, path):
    im, cb = heatmap(gates, cbar=False, cmap="binary",
                     row_labels=[f'{x + 1}' for x in range(gates.shape[0])],
                     col_labels=[f'{x + 1}' for x in range(gates.shape[1])],
                     show_axis_labels=True
                     )
    im.set_clim(0, 1)

    plt.xlabel('heads')
    plt.ylabel('layers')
    plt.savefig(os.path.join(os.path.dirname(path), 'tasks.pdf'))
    plt.show()


def main():
    cdroot()
    gates = []
    for t in ['pos', 'ner', 'dep', 'con', 'srl']:
        path = f'data/research/mtl/deberta/{t}-gates.pt'
        task_gates = torch.load(path)
        gates.append(torch.stack(task_gates).mean(0))
    draw(torch.stack(task_gates).mean(0), path)


if __name__ == '__main__':
    main()
