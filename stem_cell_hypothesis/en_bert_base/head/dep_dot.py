# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-08 22:49
import torch
import matplotlib.pyplot as plt
from elit.common.vocab import Vocab
from elit.components.mtl.gated.draw_attn import heatmap
import numpy as np
from stem_cell_hypothesis import cdroot
from stem_cell_hypothesis.en_bert_base.head.dep import DepAcc
from adjustText import adjust_text
cdroot()

rs = []
static = True
if not static:
    for i in range(3):
        save_dir = f'data/model/mtl/ontonotes_bert_base_en/pos/basic/{i}/single/records.pt'
        records = torch.load(save_dir)
        for label, count in records.label_count.items():
            records.label_correct[label] /= count
        rs.append(records)
else:
    save_dir = f'data/model/mtl/ontonotes_bert_base_en/dep/0/static/records.pt'
    records: DepAcc = torch.load(save_dir)
    records.finalize()
    rs.append(records)

records = DepAcc()
records.label_count = rs[0].label_count
ratios = dict()
dis = torch.load('data/tmp/dep_dis.pt')

total = 0
for tag, freq in rs[0].label_count.most_common():
    tag: str = tag
    # if dis[tag] < 2:
    #     continue
    # if tag in ('punct', 'num', 'dep'):
    #     continue
    # if records.label_count[tag] < 1000:
    #     continue
    # ratios[tag] = records.label_count[tag] / sum(records.label_count.values())
    # if ratios[tag] < 0.001:
    #     continue
    records.label_correct[tag] = torch.mean(torch.stack([x.label_correct[tag] for x in rs]), dim=0)
    total += 1
    # if total == 10:
    #     break

texts = []
for tag, head in records.label_correct.items():
    acc, offset = head.max(1)
    acc, layer = acc.max(0)
    acc = acc.item()
    layer = layer.item() + 1
    plt.scatter(layer, acc)
    # plt.annotate(tag, (layer, acc))
    texts.append(plt.text(layer, acc, tag))

adjust_text(texts)
plt.xticks(np.arange(1, 13))
plt.xlabel('Layers')
plt.ylabel('Accuracy')
# plt.title('Speciality of each head' + (' [static]' if static else ' [finetune]'))

if static:
    plt.savefig('data/model/mtl/ontonotes_bert_base_en/dep/0/static/dep-acc-per-layer.pdf')
else:
    plt.savefig('data/model/mtl/ontonotes_bert_base_en/dep/0/single/dep-acc-per-layer.pdf')
plt.show()
