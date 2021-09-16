# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-06 16:12
from collections import defaultdict, Counter
from typing import List

import os
import torch
import matplotlib.pyplot as plt
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.mtl.gated.draw_attn import heatmap
from elit.components.mtl.head_mtl import HeadMultiTaskLearning
from elit.components.mtl.multi_task_learning import MultiTaskLearning
from elit.components.mtl.tasks.constituency import CRFConstituencyParsing
from elit.components.mtl.tasks.dep import BiaffineDependencyParsing
from elit.components.mtl.tasks.ner.biaffine_ner import BiaffineNamedEntityRecognition
from elit.components.mtl.tasks.pos import TransformerTagging
from elit.components.mtl.tasks.srl.rank_srl import SpanRankingSemanticRoleLabeling
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_POS_ENGLISH_TRAIN, ONTONOTES5_POS_ENGLISH_TEST, \
    ONTONOTES5_POS_ENGLISH_DEV, ONTONOTES5_ENGLISH_TRAIN, ONTONOTES5_ENGLISH_TEST, ONTONOTES5_ENGLISH_DEV, \
    ONTONOTES5_CON_ENGLISH_TRAIN, ONTONOTES5_CON_ENGLISH_DEV, ONTONOTES5_CON_ENGLISH_TEST, ONTONOTES5_DEP_ENGLISH_TEST, \
    ONTONOTES5_DEP_ENGLISH_DEV, ONTONOTES5_DEP_ENGLISH_TRAIN, ONTONOTES5_NER_ENGLISH_TRAIN, ONTONOTES5_NER_ENGLISH_DEV, \
    ONTONOTES5_NER_ENGLISH_TEST, ONTONOTES5_SRL_ENGLISH_TRAIN, ONTONOTES5_SRL_ENGLISH_DEV, ONTONOTES5_SRL_ENGLISH_TEST
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from elit.metrics.mtl import MetricDict
from elit.utils.log_util import cprint
from elit.utils.time_util import CountdownTimer
from hanlp_common.constant import ROOT
from stem_cell_hypothesis import cdroot


class NerAcc(object):

    def __init__(self) -> None:
        super().__init__()
        self.label_centroid = dict()
        self.label_count = Counter()
        self.label_correct = dict()

    def finalize(self):
        for label, count in self.label_count.items():
            self.label_centroid[label] /= count


def draw_acc(acc: torch.Tensor, title):
    im, cb = heatmap(acc, cbar=True, cmap="binary",
                     row_labels=[f'{x + 1}' for x in range(acc.shape[0])],
                     col_labels=[f'{x + 1}' for x in range(acc.shape[1])],
                     show_axis_labels=True
                     )
    im.set_clim(0, 1)
    plt.xlabel('heads')
    plt.ylabel('layers')
    plt.title(title)


def main():
    cdroot()
    # save_dir = 'data/model/mtl/ontonotes_electra_base_en/all/lw/3'
    # save_dir = 'data/model/mtl/ontonotes_electra_base_en/lw/ner_srl/2'
    # save_dir = 'data/model/mtl/ontonotes_electra_base_en/lw/ner_dep/0'
    # save_dir = f'data/model/mtl/ontonotes_electra_base_en/mtl/no-gate/ner_srl/0'
    save_dir = f'data/model/mtl/ontonotes_electra_base_en/mtl/no-gate/all/2'
    # save_dir = f'data/model/mtl/ontonotes_electra_base_en/single/no-gate/ner/1'
    # folder = 'static'
    # folder = 'single'
    folder = 'joint-ner'
    print(save_dir + '/' + folder)
    records = calc_acc(save_dir, folder)
    for label, count in records.label_count.items():
        records.label_correct[label] /= count
    overall = torch.zeros((12, 12), device=records.label_correct[label].device)
    for label, count in records.label_count.items():
        overall += records.label_correct[label] * count
    overall /= sum(records.label_count.values())
    # draw_acc(overall, 'Weighted Accuracy')
    # plt.savefig(f'{save_dir}/{folder}/overall.pdf')
    # plt.clf()
    for label, freq in records.label_count.most_common():
        acc = records.label_correct[label]
        # print(f'{acc.max() * 100:.2f}')
        # print(f'{label}\t{acc.max() * 100:.2f}')
        print(f'{acc.max() * 100:.2f}')
        # draw_acc(acc, label)
        # plt.savefig(f'{save_dir}/{folder}/{label}.pdf')
        # plt.clf()


def calc_acc(save_dir, folder):
    tasks = {
        # 'pos': TransformerTagging(
        #     ONTONOTES5_POS_ENGLISH_TRAIN,
        #     ONTONOTES5_POS_ENGLISH_DEV,
        #     ONTONOTES5_POS_ENGLISH_TEST,
        #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
        #     lr=1e-3,
        # ),
        'ner': BiaffineNamedEntityRecognition(
            ONTONOTES5_NER_ENGLISH_TRAIN,
            ONTONOTES5_NER_ENGLISH_DEV,
            ONTONOTES5_NER_ENGLISH_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
            doc_level_offset=True,
        ),
        # 'srl': SpanRankingSemanticRoleLabeling(
        #     ONTONOTES5_SRL_ENGLISH_TRAIN,
        #     ONTONOTES5_SRL_ENGLISH_DEV,
        #     ONTONOTES5_SRL_ENGLISH_TEST,
        #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
        #     lr=1e-3,
        #     doc_level_offset=True,
        # ),
        # 'dep': BiaffineDependencyParsing(
        #     ONTONOTES5_DEP_ENGLISH_TRAIN,
        #     ONTONOTES5_DEP_ENGLISH_DEV,
        #     ONTONOTES5_DEP_ENGLISH_TEST,
        #     # 'data/parsing/ptb/short.conllx',
        #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
        #     lr=1e-3,
        # ),
        # 'con': CRFConstituencyParsing(
        #     ONTONOTES5_CON_ENGLISH_TRAIN,
        #     ONTONOTES5_CON_ENGLISH_DEV,
        #     ONTONOTES5_CON_ENGLISH_TEST,
        #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
        #     lr=1e-3,
        # ),
    }
    mtl = HeadMultiTaskLearning()
    # save_dir = f'data/model/mtl/ontonotes_electra_base_en/dep/0'
    path = f'{save_dir}/{folder}/records.pt'
    if os.path.isfile(path):
        return torch.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mtl.load(save_dir)
    # if 'dep' in mtl.tasks:
    #     mtl['dep'].config.tree = True
    #     mtl['dep'].config.proj = True
    # mtl.save_config(save_dir)

    for task_name in list(mtl.tasks.keys()):
        if task_name not in tasks:
            del mtl[task_name]

    for k, v in mtl.tasks.items():
        v.trn = tasks[k].trn
        v.dev = tasks[k].dev
        v.tst = tasks[k].tst
    # metric = mtl.evaluate(save_dir)[0]
    cache = mtl.dump_attended_values_per_head(save_dir, subfoler=folder)
    records = None
    timer = CountdownTimer(len(cache))
    device = mtl.device
    for batch, attentions in cache:
        if records is None:
            num_layers = len(attentions)
            num_heads = attentions[0].size(2)
            records = NerAcc()
        attn_values = torch.stack(attentions, dim=2).to(device)
        for b, ner in enumerate(batch['ner']):
            for i, (start, end, t) in enumerate(ner):
                value = attn_values[b, start:end + 1, :, :, :].mean(0)
                if t not in records.label_centroid:
                    records.label_centroid[t] = torch.zeros((num_layers, num_heads, value.size(-1)),
                                                            dtype=torch.float, device=device)
                    records.label_correct[t] = torch.zeros((num_layers, num_heads), dtype=torch.float, device=device)
                records.label_centroid[t] += value
                records.label_count[t] += 1
        timer.log()

    records.finalize()
    centroids = torch.stack(tuple(records.label_centroid.values()))
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    vocabs = dict((x, i) for i, x in enumerate(records.label_centroid.keys()))
    timer = CountdownTimer(len(cache))
    for batch, attentions in cache:
        attn_values = torch.stack(attentions, dim=2).to(device)
        for b, ner in enumerate(batch['ner']):
            for i, (start, end, t) in enumerate(ner):
                value = attn_values[b, start:end + 1, :, :, :].mean(0)
                sim = cos(value.expand_as(centroids), centroids)
                closest = sim.argmax(0)
                correct = closest == vocabs[t]
                records.label_correct[t] += correct
        timer.log()

    torch.save(records, path)
    return records


if __name__ == '__main__':
    main()
