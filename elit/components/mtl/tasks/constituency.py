# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-29 16:52
import logging
from typing import Dict, Any, List, Union, Iterable, Callable

import torch
import torch.nn.functional as F
from phrasetree.tree import Tree

from elit.components.distillation.losses import kd_mse_loss
from hanlp_common.constant import BOS, EOS
from hanlp_common.document import Document
from elit.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict
from elit.components.mtl.tasks import Task, TemperatureFunctionType, KDLossFunctionType
from elit.components.parsers.constituency.crf_constituency_model import CRFConstituencyDecoder
from elit.components.parsers.constituency.crf_constituency_parser import CRFConstituencyParser
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, prefix_match


class CRFConstituencyParsing(Task, CRFConstituencyParser):
    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=None,
                 separate_optimizer=False,
                 cls_is_bos=True,
                 sep_is_eos=True,
                 delete=('', ':', '``', "''", '.', '?', '!', '-NONE-', 'TOP', ',', 'S1'),
                 equal=(('ADVP', 'PRT'),),
                 mbr=True,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 no_subcategory=True,
                 max_seq_len=None,
                 **kwargs
                 ) -> None:
        r"""Two-stage CRF Parsing (:cite:`ijcai2020-560`).

        Args:
            trn: Path to training set.
            dev: Path to dev set.
            tst: Path to test set.
            sampler_builder: A builder which builds a sampler.
            dependencies: Its dependencies on other tasks.
            scalar_mix: A builder which builds a `ScalarMixWithDropout` object.
            use_raw_hidden_states: Whether to use raw hidden states from transformer without any pooling.
            lr: Learning rate for this task.
            separate_optimizer: Use customized separate optimizer for this task.
            cls_is_bos: ``True`` to treat the first token as ``BOS``.
            sep_is_eos: ``True`` to treat the last token as ``EOS``.
            delete: Constituencies to be deleted from training and evaluation.
            equal: Constituencies that are regarded as equal during evaluation.
            mbr: ``True`` to enable Minimum Bayes Risk (MBR) decoding (:cite:`smith-smith-2007-probabilistic`).
            n_mlp_span: Number of features for span decoder.
            n_mlp_label: Number of features for label decoder.
            mlp_dropout: Dropout applied to MLPs.
            no_subcategory: Strip out subcategories.
            **kwargs: Not used.
        """
        if isinstance(equal, tuple):
            equal = dict(equal)
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    # noinspection DuplicatedCode
    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        dataset = CRFConstituencyParsing.build_dataset(self, data, transform)
        if isinstance(data, str):
            dataset.purge_cache()
        if self.vocabs.mutable:
            CRFConstituencyParsing.build_vocabs(self, dataset, logger)
        if isinstance(data, str):
            max_seq_len = self.config.get('max_seq_len', None)
            if max_seq_len:
                dataset.prune(lambda x: len(x['token_input_ids']) > max_seq_len, logger)
        if dataset.cache:
            timer = CountdownTimer(len(dataset))
            # noinspection PyCallByClass
            BiaffineDependencyParser.cache_dataset(self, dataset, timer, training, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def feed_batch(self,
                   h: torch.FloatTensor,
                   batch: Dict[str, torch.Tensor],
                   mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        return {
            'output': decoder(h),
            'mask': CRFConstituencyParser.compute_mask(
                self, batch, offset=1 if 'constituency' in batch or batch['token'][0][-1] == EOS else -1)
        }

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        out, mask = output['output'], output['mask']
        loss, span_probs = CRFConstituencyParser.compute_loss(self, out, batch['chart_id'], mask, crf_decoder=criterion)
        output['span_probs'] = span_probs
        return loss

    def compute_kd_loss(self, batch: Dict[str, Any],
                        output_student: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                        output_teacher: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                        temperature_function: TemperatureFunctionType,
                        kd_loss_function: KDLossFunctionType) -> torch.FloatTensor:
        (span_s, label_s), mask_s = output_student['output'], output_student['mask']
        (span_t, label_t), mask_t = output_teacher['output'], output_teacher['mask']
        assert mask_s.equal(mask_t), 'Student output mismatches with teacher output, batching went wrong'
        span_s = span_s[mask_s]
        span_t = span_t[mask_t]
        label_s = label_s[mask_s]
        label_t = label_t[mask_t]
        span_loss = kd_mse_loss(span_s, span_t, temperature_function(span_s, span_t))
        label_loss = kd_loss_function(label_s, label_t, temperature_function(label_s, label_t))
        return span_loss + label_loss

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder: torch.nn.Module, **kwargs) -> Union[Dict[str, Any], Any]:
        out, mask = output['output'], output['mask']
        tokens = []
        for sent in batch['token']:
            if sent[0] == BOS:
                sent = sent[1:]
            if sent[-1] == EOS:
                sent = sent[:-1]
            tokens.append(sent)
        return CRFConstituencyParser.decode_output(self, out, mask, batch, output.get('span_probs', None),
                                                   decoder=decoder, tokens=tokens)

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        return CRFConstituencyParser.update_metrics(self, metric, batch, prediction)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return CRFConstituencyDecoder(n_labels=len(self.vocabs.chart), n_hidden=encoder_size)

    def build_metric(self, **kwargs):
        return CRFConstituencyParser.build_metric(self)

    def input_is_flat(self, data) -> bool:
        return CRFConstituencyParser.input_is_flat(self, data)

    def prediction_to_result(self, prediction: List, batch: Dict[str, Any]) -> List:
        return prediction

    def finalize_document(self, doc: Document, task_name: str):
        pos_key = prefix_match('pos', doc)
        pos: List[List[str]] = doc.get(pos_key, None)
        if pos:
            for tree, pos_per_sent in zip(doc[task_name], pos):
                tree: Tree = tree
                offset = 0
                for subtree in tree.subtrees(lambda t: t.height() == 2):
                    tag = subtree.label()
                    if tag == '_':
                        subtree.set_label(pos_per_sent[offset])
                    offset += 1

    def build_samples(self, inputs, cls_is_bos=False, sep_is_eos=False):
        return CRFConstituencyParser.build_samples(self, inputs)
