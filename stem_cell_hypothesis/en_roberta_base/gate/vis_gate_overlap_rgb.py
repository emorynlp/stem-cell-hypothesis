# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-08 17:51
from typing import List
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import MaxNLocator

from elit.components.mtl.gated.gated_mtl import GatedMultiTaskLearning
from stem_cell_hypothesis import cdroot

cdroot()

# cache_path = 'data/research/mtl/roberta/dep-gates.pt'
# cache_path = 'data/research/mtl/roberta/ner-gates.pt'
# cache_path = 'data/research/mtl/roberta/con-gates.pt'
# cache_path = 'data/research/mtl/roberta/srl-gates.pt'
# cache_path = 'data/research/mtl/roberta/pos-gates.pt'
cache_path = 'data/research/mtl/roberta/joint-gates.pt'

# gates = []
# for i in range(3):
#     path = f'data/model/mtl/ontonotes_roberta_base_en/gate/all/{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/gated/pos/finetune/encoder1e-5_{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/single/gated/dep/coef0.01_epochs30/{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/single/gated/ner/{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/single/gate/con/{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/single/gated/srl/{i}'
#     # path = f'data/model/mtl/ontonotes_roberta_base_en/single/gated/pos/{i}'
#     mtl = GatedMultiTaskLearning()
#     mtl.load(path, devices=-1)
#     gates.append(mtl.get_gates())
# torch.save(gates, cache_path)
gates = torch.load(cache_path)


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

            # Create a Rectangle patch
            rect = patches.Rectangle((j - 0.5, i - 0.5),
                                     1, 1, linewidth=0,
                                     facecolor=tuple(1 - gates[k][i][j].item() for k in range(len(gates))))
            # Add the patch to the Axes
            ax.add_patch(rect)

    if 'joint' in cache_path:
        ax.set_xticks(np.arange(12))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([f'{i + 1}' for i in range(12)])
        ax.set_yticklabels([f'{i + 1}' for i in range(12)])
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.xlabel('Heads')
        plt.ylabel('Layers')
    else:
        ax.tick_params(left=False, top=False, bottom=False, labeltop=False, labelbottom=False, labelleft=False)
    plt.savefig(cache_path.replace('.pt', '.pdf'))
    plt.show()


def main():
    gates = torch.load(cache_path)
    plt.rcParams.update({'font.size': 12})
    draw(gates)


if __name__ == '__main__':
    main()
