# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-11 19:36
from collections import Counter, defaultdict

import torch

from elit.datasets.parsing.conll_dataset import CoNLLParsingDataset
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_DEP_ENGLISH_TRAIN
from stem_cell_hypothesis import cdroot

cdroot()
dataset = CoNLLParsingDataset(ONTONOTES5_DEP_ENGLISH_TRAIN)

distance = defaultdict(Counter)
for sample in dataset:
    for i, (h, r) in enumerate(zip(sample['HEAD'], sample['DEPREL'])):
        offset = i + 1
        distance[r][abs(h - offset)] += 1

dis = dict()
for rel, c in distance.items():
    dis[rel] = sum(dis * f for (dis, f) in c.items()) / sum(c.values())
    # print(f'{rel} {sum(d * f for (d, f) in c.items()) / sum(c.values()):.1f}')

for rel, d in sorted(dis.items(), key=lambda x: x[1], reverse=True):
    print(f'{rel} {d:.1f}')

torch.save(dis, 'data/research/mtl/deberta/dep_dis.pt')

