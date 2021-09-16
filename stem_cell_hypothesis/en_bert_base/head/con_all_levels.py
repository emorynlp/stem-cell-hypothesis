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
from stem_cell_hypothesis.en_bert_base.head.debug_con_visualize import offset_of, extract_spans


class ConAcc(object):

    def __init__(self) -> None:
        super().__init__()
        self.label_correct = dict()
        self.label_count = Counter()

    def finalize(self):
        for label, count in self.label_count.items():
            self.label_correct[label] = self.label_correct.get(label, 0) / count


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
    # save_dir = 'data/model/mtl/ontonotes_bert_base_en/all/lw/1'
    save_dir = 'data/model/mtl/ontonotes_bert_base_en/con/1'
    # folder = 'joint-con'
    folder = 'single'
    records = calc_acc(save_dir, folder)
    records.finalize()
    overall = torch.zeros((12, 12))
    for label, count in records.label_count.items():
        overall += records.label_correct[label] * count
    overall /= sum(records.label_count.values())
    # draw_acc(overall, 'Weighted Accuracy')
    # plt.savefig(f'{save_dir}/{folder}/overall.pdf')
    # plt.clf()
    num_items = 0
    labels = ['PP-4', 'VP-4', 'ADVP-3', 'VP-3', 'NML-3', 'PP-5', 'WHNP-3', 'S-4', 'QP-3', 'WHADVP-3', 'PRT-3', 'SBAR-5',
              'PRN-5', 'ADVP-5', 'FRAG-4', 'PP-3', 'ADVP-4', 'SQ-4', 'FRAG-5', 'CONJP-3', 'EMBED-4']
    for label, freq in records.label_count.most_common():
        acc = records.label_correct.get(label, 0)
        if isinstance(acc, torch.Tensor):
            acc = acc.max()
        # if acc < .5:
        #     continue
        if label not in labels:
            continue
        print(f'{acc.max() * 100:.2f}')
        # print(f'{label}\t{acc * 100:.2f}')
        num_items += 1
        if num_items > 20:
            break
        # print(f'{acc.max() * 100:.2f}')
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
        # 'ner': BiaffineNamedEntityRecognition(
        #     ONTONOTES5_NER_ENGLISH_TRAIN,
        #     ONTONOTES5_NER_ENGLISH_DEV,
        #     ONTONOTES5_NER_ENGLISH_TEST,
        #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
        #     lr=1e-3,
        #     doc_level_offset=True,
        # ),
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
        'con': CRFConstituencyParsing(
            ONTONOTES5_CON_ENGLISH_TRAIN,
            ONTONOTES5_CON_ENGLISH_DEV,
            ONTONOTES5_CON_ENGLISH_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
        ),
    }
    mtl = HeadMultiTaskLearning()
    # save_dir = f'data/model/mtl/ontonotes_bert_base_en/dep/0'
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
    cache = mtl.dump_attention_per_head(save_dir, subfoler=folder)
    records = None
    timer = CountdownTimer(len(cache))
    for batch, attentions in cache:
        if records is None:
            num_layers = len(attentions)
            num_heads = attentions[0].size(1)
            records = ConAcc()
        for layer, attn_per_layer in enumerate(attentions):
            max_attn = attn_per_layer.argmax(dim=-1)
            for head in range(max_attn.size(1)):
                attn = max_attn[:, head, :]
                for b, con in enumerate(batch['constituency']):
                    for begin, end, label in extract_spans(con, offset=1):
                        if attn[b][begin].item() == end or attn[b][end].item() == begin:
                            correct = records.label_correct.get(label, None)
                            if correct is None:
                                # noinspection PyUnboundLocalVariable
                                correct = records.label_correct[label] = torch.zeros([num_layers, num_heads])
                            correct[layer][head] += 1
                        if layer == 0 and head == 0:
                            records.label_count[label] += 1
        timer.log()
    torch.save(records, path)
    return records


if __name__ == '__main__':
    main()
