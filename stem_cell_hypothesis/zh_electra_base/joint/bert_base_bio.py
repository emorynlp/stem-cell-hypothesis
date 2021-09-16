# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-06 16:12
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.mtl.multi_task_learning import MultiTaskLearning
from elit.components.mtl.tasks.constituency import CRFConstituencyParsing
from elit.components.mtl.tasks.dep import BiaffineDependencyParsing
from elit.components.mtl.tasks.ner.biaffine_ner import BiaffineNamedEntityRecognition
from elit.components.mtl.tasks.ner.tag_ner import TaggingNamedEntityRecognition
from elit.components.mtl.tasks.pos import TransformerTagging
from elit.components.mtl.tasks.srl.rank_srl import SpanRankingSemanticRoleLabeling
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.chinese import ONTONOTES5_POS_CHINESE_TRAIN, ONTONOTES5_POS_CHINESE_TEST, \
    ONTONOTES5_POS_CHINESE_DEV, ONTONOTES5_CHINESE_TRAIN, ONTONOTES5_CHINESE_TEST, ONTONOTES5_CHINESE_DEV, \
    ONTONOTES5_CON_CHINESE_TRAIN, ONTONOTES5_CON_CHINESE_DEV, ONTONOTES5_CON_CHINESE_TEST, ONTONOTES5_DEP_CHINESE_TEST, \
    ONTONOTES5_DEP_CHINESE_DEV, ONTONOTES5_DEP_CHINESE_TRAIN, ONTONOTES5_NER_CHINESE_TSV_TRAIN, ONTONOTES5_NER_CHINESE_TSV_DEV, \
    ONTONOTES5_NER_CHINESE_TSV_TEST
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from elit.layers.transformers.relative_transformer import RelativeTransformerEncoder
from elit.utils.log_util import cprint
from stem_cell_hypothesis import cdroot


def main():
    cdroot()
    tasks = {
        'pos': TransformerTagging(
            ONTONOTES5_POS_CHINESE_TRAIN,
            ONTONOTES5_POS_CHINESE_DEV,
            ONTONOTES5_POS_CHINESE_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
        ),
        'ner': TaggingNamedEntityRecognition(
            ONTONOTES5_NER_CHINESE_TSV_TRAIN,
            ONTONOTES5_NER_CHINESE_TSV_DEV,
            ONTONOTES5_NER_CHINESE_TSV_TEST,
            SortingSamplerBuilder(batch_size=32),
            lr=1e-3,
            secondary_encoder=RelativeTransformerEncoder(768, k_as_x=True),
        ),
        'srl': SpanRankingSemanticRoleLabeling(
            ONTONOTES5_CHINESE_TRAIN,
            ONTONOTES5_CHINESE_DEV,
            ONTONOTES5_CHINESE_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
            doc_level_offset=True,
        ),
        'dep': BiaffineDependencyParsing(
            ONTONOTES5_DEP_CHINESE_TRAIN,
            ONTONOTES5_DEP_CHINESE_DEV,
            ONTONOTES5_DEP_CHINESE_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
        ),
        'con': CRFConstituencyParsing(
            ONTONOTES5_CON_CHINESE_TRAIN,
            ONTONOTES5_CON_CHINESE_DEV,
            ONTONOTES5_CON_CHINESE_TEST,
            SortingSamplerBuilder(batch_size=64, batch_max_tokens=6400),
            lr=1e-3,
        ),
    }
    mtl = MultiTaskLearning()
    save_dir = 'data/model/mtl/ontonotes_electra_base_zh_2'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    mtl.fit(
        ContextualWordEmbedding(
            'token',
            "hfl/chinese-electra-180g-base-discriminator",
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
        gradient_accumulation=4,
        eval_trn=False,

        prefetch=10,
        cache='data/tmp'
    )
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    mtl.load(save_dir)
    mtl['dep'].config.tree = True
    mtl['dep'].config.proj = True
    mtl.save_config(save_dir)
    for k, v in mtl.tasks.items():
        v.trn = tasks[k].trn
        v.dev = tasks[k].dev
        v.tst = tasks[k].tst
    mtl.evaluate(save_dir)
    mtl([['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'],
         ['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']]).pretty_print()


if __name__ == '__main__':
    import torch

    torch.multiprocessing.set_start_method('spawn')  # See https://github.com/pytorch/pytorch/issues/40403
    main()
