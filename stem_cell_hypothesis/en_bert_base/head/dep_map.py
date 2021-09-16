# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-08 22:49
import torch
import matplotlib.pyplot as plt
from elit.common.vocab import Vocab
from elit.components.mtl.gated.draw_attn import heatmap

from stem_cell_hypothesis import cdroot
from stem_cell_hypothesis.en_bert_base.head.dep import DepAcc

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
vocab = Vocab(pad_token=None, unk_token=None)
ratios = dict()
dis = torch.load('data/tmp/dep_dis.pt')

for tag in rs[0].label_correct:
    tag: str = tag
    if dis[tag] < 2:
        continue
    if tag in ('punct', 'num'):
        continue
    if records.label_count[tag] < 1000:
        continue
    # ratios[tag] = records.label_count[tag] / sum(records.label_count.values())
    # if ratios[tag] < 0.001:
    #     continue
    records.label_correct[tag] = torch.mean(torch.stack([x.label_correct[tag] for x in rs]), dim=0)
    vocab.add(tag)

vocab.lock()
vocab.summary()

heads, indices = torch.stack(list(records.label_correct.values())).max(0)
cell_labels = []
for j in range(indices.shape[1]):
    cell_labels.append([vocab.idx_to_token[i] for i in indices[j, :]])
    # cell_labels.append([f'{dis[vocab.idx_to_token[i]]:.1f}' for i in indices[j, :]])
# heads[heads < 0.6] = 0
im, cb = heatmap(heads, cbar=True, cmap="binary",
                 row_labels=[f'{x + 1}' for x in range(heads.shape[0])],
                 col_labels=[f'{x + 1}' for x in range(heads.shape[1])],
                 show_axis_labels=True,
                 cell_labels=cell_labels
                 )
im.set_clim(0, 1)
plt.xlabel('heads')
plt.ylabel('layers')
# plt.title('Speciality of each head' + (' [static]' if static else ' [finetune]'))

if static:
    plt.savefig('data/model/mtl/ontonotes_bert_base_en/dep/0/static/speciality.pdf')
else:
    plt.savefig('data/model/mtl/ontonotes_bert_base_en/dep/0/single/speciality.pdf')
plt.show()
