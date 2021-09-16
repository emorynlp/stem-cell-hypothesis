# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-08 22:49
import matplotlib.pyplot as plt
import torch

from stem_cell_hypothesis import cdroot
from stem_cell_hypothesis.en_bert_base.head.srl import SrlAcc
import numpy as np
from tests import cdroot
from stem_cell_hypothesis.en_bert_base.head.dep import DepAcc
from adjustText import adjust_text
cdroot()

rs = []
static = True
if not static:
    for i in range(3):
        save_dir = f'data/model/mtl/ontonotes_roberta_base_en/pos/basic/{i}/single/records.pt'
        records = torch.load(save_dir)
        for label, count in records.label_count.items():
            records.label_correct[label] /= count
        rs.append(records)
else:
    save_dir = f'data/model/mtl/ontonotes_roberta_base_en_/srl/0/static/records.pt'
    records: SrlAcc = torch.load(save_dir)
    records.finalize()
    rs.append(records)

records = SrlAcc()
records.label_count = rs[0].label_count
ratios = dict()

total = 0
for tag, freq in rs[0].label_count.most_common():
    tag: str = tag
    # if tag in ('punct', 'num'):
    #     continue
    # if records.label_count[tag] < 1000:
    #     continue
    # ratios[tag] = records.label_count[tag] / sum(records.label_count.values())
    # if ratios[tag] < 0.001:
    #     continue
    records.label_correct[tag] = torch.mean(torch.stack([x.label_correct[tag] for x in rs]), dim=0)
    total += 1
    if total == 30:
        break

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
# plt.xticks(np.arange(1, 13))

# plt.legend(loc='upper right')
plt.xlabel('Layers')
plt.ylabel('Accuracy')
# plt.title('Speciality of each head' + (' [static]' if static else ' [finetune]'))

if static:
    plt.savefig('data/model/mtl/ontonotes_roberta_base_en_/srl/0/static/srl-acc-per-layer.pdf')
else:
    plt.savefig('data/model/mtl/ontonotes_roberta_base_en_/srl/0/single/srl-acc-per-layer.pdf')
plt.show()
