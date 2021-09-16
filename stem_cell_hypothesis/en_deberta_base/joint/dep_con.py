# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-06 16:12
from typing import List

from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.mtl.loss_balancer import MovingAverageBalancer
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
    ONTONOTES5_DEP_ENGLISH_DEV, ONTONOTES5_DEP_ENGLISH_TRAIN
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from elit.metrics.mtl import MetricDict
from elit.utils.log_util import cprint
from stem_cell_hypothesis import cdroot


def main():
    cdroot()
    scores: List[MetricDict] = []
    for i in range(3):
        tasks = {
            # 'pos': TransformerTagging(
            #     ONTONOTES5_POS_ENGLISH_TRAIN,
            #     ONTONOTES5_POS_ENGLISH_DEV,
            #     ONTONOTES5_POS_ENGLISH_TEST,
            #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            #     lr=1e-3,
            # ),
            # 'ner': BiaffineNamedEntityRecognition(
            #     ONTONOTES5_ENGLISH_TRAIN,
            #     ONTONOTES5_ENGLISH_DEV,
            #     ONTONOTES5_ENGLISH_TEST,
            #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            #     lr=1e-3,
            #     doc_level_offset=True,
            # ),
            # 'srl': SpanRankingSemanticRoleLabeling(
            #     ONTONOTES5_ENGLISH_TRAIN,
            #     ONTONOTES5_ENGLISH_DEV,
            #     ONTONOTES5_ENGLISH_TEST,
            #     SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            #     lr=1e-3,
            #     doc_level_offset=True,
            # ),
            'dep': BiaffineDependencyParsing(
                ONTONOTES5_DEP_ENGLISH_TRAIN,
                ONTONOTES5_DEP_ENGLISH_DEV,
                ONTONOTES5_DEP_ENGLISH_TEST,
                SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
                lr=1e-3,
            ),
            'con': CRFConstituencyParsing(
                ONTONOTES5_CON_ENGLISH_TRAIN,
                ONTONOTES5_CON_ENGLISH_DEV,
                ONTONOTES5_CON_ENGLISH_TEST,
                SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
                lr=1e-3,
            ),
        }
        mtl = MultiTaskLearning()
        save_dir = f'data/model/mtl/ontonotes_deberta_base_en/pairwise/dep_con/lw/{i}'
        cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
        mtl.fit(
            ContextualWordEmbedding(
                'token',
                'microsoft/deberta-base',
                average_subwords=True,
                max_sequence_length=512,
                word_dropout=.2,
            ),
            tasks,
            save_dir,
            30,
            lr=1e-3,
            encoder_lr=5e-5,
            grad_norm=1,
            gradient_accumulation=1,
            eval_trn=False,
            transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
            loss_balancer=MovingAverageBalancer(5, intrinsic_weighting=False),
            # prefetch=10,
            # cache='data/tmp'
        )
        cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
        mtl.load(save_dir)
        if 'dep' in mtl.tasks:
            mtl['dep'].config.tree = True
            mtl['dep'].config.proj = True
        mtl.save_config(save_dir)
        for k, v in mtl.tasks.items():
            v.trn = tasks[k].trn
            v.dev = tasks[k].dev
            v.tst = tasks[k].tst
        metric = mtl.evaluate(save_dir)[0]
        scores.append(metric)

    print(f'{"-".join(tasks.keys())} {len(scores)} runs scores:')
    for each in scores:
        cprint(each.cstr())


if __name__ == '__main__':
    import torch

    # torch.multiprocessing.set_start_method('spawn')  # See https://github.com/pytorch/pytorch/issues/40403
    main()
