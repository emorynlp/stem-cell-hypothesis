# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-08 17:51
from typing import List

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from statsmodels.formula.api import ols

from stem_cell_hypothesis import cdroot

cdroot()


def pearsonr(x, y, batch_first=True):
    # https://github.com/audeering/audtorch/blob/0.1.1/audtorch/metrics/functional.py
    r"""Computes Pearson Correlation Coefficient across rows.
    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:
    .. math::
        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}
    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.
    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


def pearson_cos(x1, x2, dim=1):
    # https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/10
    cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    pearson = cos(x1 - x1.mean(dim=dim, keepdim=True), x2 - x2.mean(dim=dim, keepdim=True))
    return pearson


def main():
    tasks = ['pos', 'ner', 'dep', 'con', 'srl', 'joint']
    matrix = torch.zeros((len(tasks), len(tasks)))
    for t1 in range(len(tasks)):
        task_1 = tasks[t1]
        gates1 = torch.load(f'data/tmp/bert-base-{task_1}-gates.pt')
        for t2 in range(t1 + 1, len(tasks)):
            task_2 = tasks[t2]
            gates2 = torch.load(f'data/tmp/bert-base-{task_2}-gates.pt')
            v1 = torch.stack(gates1).mean(0).reshape(-1)
            v2 = torch.stack(gates2).mean(0).reshape(-1)
            p = pearsonr(v1, v2)
            # Alternatively use cos similarity
            # pearson_cos(v1.unsqueeze(0), v2.unsqueeze(0))
            matrix[t1][t2] = p
            matrix[t2][t1] = p
        nlayer, nhead = gates1[0].size()
        data = []
        for i in range(nlayer):
            for j in range(nhead):
                z1 = gates1[0][i][j].item()
                z2 = gates1[1][i][j].item()
                z3 = gates1[2][i][j].item()
                data.append([f'Layer{i}-Head{j}', z1, z2, z3])
        data = pandas.DataFrame(data,
                                columns=["ID", "z1", "z2", "z3"])
        model = ols("z3 ~ z1 + z2", data=data).fit()
        r = model.rsquared_adj ** .5
        matrix[t1][t1] = r
    # tasks = [x.upper() for x in tasks]
    # tasks[-1] = 'MTL-5'
    tasks = '''
    \\bf \\POS
\\bf \\NER
\\bf \\DEP
\\bf \\CON
\\bf \\SRL
\\bf MTL-5
    '''.splitlines()
    tasks = [x.strip() for x in tasks]
    tasks = [x for x in tasks if x]
    print(' \t' + '\t'.join(tasks))
    for t1 in range(len(tasks)):
        task_1 = tasks[t1]
        row = [task_1]
        for t2 in range(len(tasks)):
            task_2 = tasks[t2]
            if t1 > t2 and False:
                row.append('-')
            else:
                row.append(f'{matrix[t1][t2] * 100:.2f}')
            if t1 == t2:
                row[-1] = '\\cellcolor{gray!32}' + row[-1]
        print(' & '.join(row) + ' \\\\')


if __name__ == '__main__':
    main()
