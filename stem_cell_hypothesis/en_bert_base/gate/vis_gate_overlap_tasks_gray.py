# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-08 17:51
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


def draw(gates: torch.Tensor):
    # Create figure and axes
    h, w = gates.shape
    fig, ax = plt.subplots(figsize=(w / 2, h / 2))
    ax.imshow(np.ones((h, w, 3)))

    for i in range(h):
        for j in range(w):
            ax.add_patch(
                patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.1, facecolor='none', edgecolor='black'))

            # Create a Rectangle patch
            rect = patches.Rectangle((j - 0.5, i - 0.5),
                                     1, 1, linewidth=0,
                                     facecolor=(0, 0, 0, gates[i][j].item()))
            # Add the patch to the Axes
            ax.add_patch(rect)

    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels([f'{i + 1}' for i in range(12)])
    ax.set_yticklabels([f'{i + 1}' for i in range(12)])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xlabel('Heads')
    plt.ylabel('Layers')


def main():
    cdroot()
    gates = []
    for t in ['pos', 'ner', 'dep', 'con', 'srl']:
        path = f'data/research/mtl/bert/{t}-gates.pt'
        task_gates = torch.load(path)
        gates.append(torch.stack(task_gates).mean(0))
    plt.rcParams.update({'font.size': 12})
    draw(torch.stack(task_gates).mean(0))
    plt.savefig('data/research/mtl/bert/tasks.pdf')
    plt.show()


if __name__ == '__main__':
    main()
