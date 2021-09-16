# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-08 17:51
from typing import List
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from elit.components.mtl.gated.gated_mtl import GatedMultiTaskLearning
from stem_cell_hypothesis import cdroot

cdroot()

gates = []
for t in ['pos', 'ner', 'dep', 'con', 'srl']:
    path = f'data/mtl/gates/{t}'
    mtl = GatedMultiTaskLearning()
    mtl.load(path, devices=-1)
    gates.append(mtl.get_gates())
torch.save(gates, 'data/tmp/gates_tasks.pt')


def draw(gates: List[torch.Tensor]):
    # Create figure and axes
    h, w = gates[0].shape
    fig, ax = plt.subplots(figsize=(w / 2, h / 2))
    ax.imshow(np.ones((h, w, 3)))
    colormap = []
    for name in 'red', 'green', 'blue', 'orange', 'purple':
        colormap.append(mcolors.to_rgb(name))
    # for k, g in enumerate(gates):
    #     p = ax.imshow(g, cmap=colormap[k])
    #     cb = plt.colorbar(p, shrink=0.25)

    for i in range(h):
        for j in range(w):
            ax.add_patch(
                patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.1, facecolor='none', edgecolor='black'))
            gs = []
            for k in range(len(gates)):
                if gates[k][i][j] > 1e-3:
                    gs.append(k)

            for l, k in enumerate(gs):
                # Create a Rectangle patch
                rect = patches.Rectangle((j + l / len(gs) - 0.5, i - 0.5),
                                         1 / len(gs), 1, linewidth=0,
                                         facecolor=colormap[k] + (gates[k][i][j],))
                # Add the patch to the Axes
                ax.add_patch(rect)

    plt.xlabel('heads')
    plt.ylabel('layers')
    plt.savefig('tasks.png')
    plt.show()


def main():
    gates = torch.load('data/tmp/gates_tasks.pt')
    draw(gates)


if __name__ == '__main__':
    main()
