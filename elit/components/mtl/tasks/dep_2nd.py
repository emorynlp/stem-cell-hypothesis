# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-07 14:14
import logging
from collections import Counter
from typing import Dict, Any, Union, Iterable, Callable, List

import torch
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict
from elit.components.mtl.tasks import Task
from elit.components.parsers.biaffine.biaffine_2nd_dep import BiaffineSecondaryParser, BiaffineJointDecoder, \
    BiaffineSeparateDecoder
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.time_util import CountdownTimer
from hanlp_common.conll import CoNLLSentence, CoNLLUWord
from hanlp_common.util import merge_locals_kwargs
from alnlp.modules import util


class BiaffineSecondaryDependencyDecoder(torch.nn.Module):
    def __init__(self, hidden_size, config) -> None:
        super().__init__()
        self.decoder = BiaffineJointDecoder(hidden_size, config) if config.joint \
            else BiaffineSeparateDecoder(hidden_size, config)

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        if mask is None:
            mask = util.lengths_to_mask(batch['token_length'])
        else:
            mask = mask.clone()
        scores = self.decoder(contextualized_embeddings, mask)
        mask[:, 0] = 0
        return scores, mask


class BiaffineSecondaryDependencyParsing(Task, BiaffineSecondaryParser):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=2e-3, separate_optimizer=False,
                 punct=False,
                 tree=False,
                 apply_constraint=True,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 pad_rel=None,
                 joint=True,
                 mu=.9,
                 nu=.9,
                 epsilon=1e-12,
                 cls_is_bos=True,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        dataset = BiaffineSecondaryParser.build_dataset(self, data, transform)
        if isinstance(data, str):
            dataset.purge_cache()
        if self.vocabs.mutable:
            BiaffineSecondaryParser.build_vocabs(self, dataset, logger, transformer=True)
        max_seq_len = self.config.get('max_seq_len', None)
        if max_seq_len and isinstance(data, str):
            dataset.prune(lambda x: len(x['token_input_ids']) > 510, logger)
        if dataset.cache:
            timer = CountdownTimer(len(dataset))
            self.cache_dataset(dataset, timer, training, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset,
            pad={'arc': 0, 'arc_2nd': False})

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):

        BiaffineSecondaryParser.update_metric(self, *prediction, batch['arc'], batch['rel_id'], output[1],
                                              batch['punct_mask'], metric, batch)

    def decode_output(self, output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor, batch: Dict[str, Any], decoder: torch.nn.Module, **kwargs) -> Union[
        Dict[str, Any], Any]:
        return BiaffineSecondaryParser.decode(self, *output[0], output[1], batch=batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return BiaffineSecondaryParser.compute_loss(self, *output[0], batch['arc'], batch['rel_id'], output[1],
                                                    criterion, batch)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineSecondaryDependencyDecoder(encoder_size, self.config)

    def build_metric(self, **kwargs):
        return BiaffineSecondaryParser.build_metric(self, **kwargs)

    def build_criterion(self, **kwargs):
        return BiaffineSecondaryParser.build_criterion(self, **kwargs)

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        config = self.config
        optimizer = torch.optim.Adam(decoder.parameters(),
                                     config.lr,
                                     (config.mu, config.nu),
                                     config.epsilon)
        return optimizer

    def input_is_flat(self, data) -> bool:
        return BiaffineSecondaryParser.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        outputs = []
        BiaffineSecondaryParser.predictions_to_human(self, prediction, outputs, batch['token'], use_pos=False)
        for sent in outputs:
            head_rel_pairs_per_sent = []
            sent: CoNLLSentence = sent
            for word in sent:
                head_rel_pairs_per_word = []
                word: CoNLLUWord = word
                head_rel_pairs_per_word.append((word.head, word.deprel))
                if word.deps:
                    head_rel_pairs_per_word += word.deps
                head_rel_pairs_per_sent.append(head_rel_pairs_per_word)
            yield head_rel_pairs_per_sent

    def cache_dataset(self, dataset, timer, training=False, logger=None):
        if not self.config.apply_constraint:
            for each in dataset:
                timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
            return
        num_roots = Counter()
        no_zero_head = True
        root_rels = Counter()
        for each in dataset:
            if training:
                num_roots[sum([x[0] for x in each['arc_2nd']])] += 1
                no_zero_head &= all([x != '_' for x in each['DEPS']])
                root_rels.update([r for a, r in list(zip(each['arc'], each['rel']))[1:] if a == 0])
            timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
        if training:
            if self.config.get('single_root', None) is None:
                self.config.single_root = len(num_roots) == 1 and num_roots.most_common()[0][0] == 1
            if self.config.get('no_zero_head', None) is None:
                self.config.no_zero_head = no_zero_head
            root_rel = root_rels.most_common()[0][0]
            self.config.root_rel_id = self.vocabs['rel'].get_idx(root_rel)
            if logger:
                logger.info(f'Training set properties: [blue]single_root = {self.config.single_root}[/blue], '
                            f'[blue]no_zero_head = {no_zero_head}[/blue], '
                            f'[blue]root_rel = {root_rel}[/blue]')
