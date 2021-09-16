# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-05 15:43
import logging
from typing import Union, List, Dict, Any, Iterable, Callable

import torch
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict
from elit.components.distillation.losses import kd_mse_loss
from elit.components.mtl.tasks import Task, TemperatureFunctionType, KDLossFunctionType
from elit.components.srl.span_rank.span_rank import SpanRankingSemanticRoleLabeler
from elit.components.srl.span_rank.span_ranking_srl_model import SpanRankingSRLDecoder
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.metrics.srl.e2e_srl import SemanticRoleLabelingMetrics
from hanlp_common.util import merge_locals_kwargs


class SpanRankingSemanticRoleLabeling(Task, SpanRankingSemanticRoleLabeler):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=1e-3, separate_optimizer=False,
                 lexical_dropout=0.5,
                 dropout=0.2,
                 span_width_feature_size=20,
                 ffnn_size=150,
                 ffnn_depth=2,
                 argument_ratio=0.8,
                 predicate_ratio=0.4,
                 max_arg_width=30,
                 mlp_label_size=100,
                 max_seq_len=None,
                 enforce_srl_constraint=False,
                 use_gold_predicates=False,
                 doc_level_offset=True,
                 use_biaffine=False,
                 loss_reduction='mean',
                 with_argument=' ',
                 **kwargs) -> None:
        r""" An implementation of "Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling"
        (:cite:`he-etal-2018-jointly`). It generates candidates triples of (predicate, arg_start, arg_end) and rank them.

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
            lexical_dropout: Dropout applied to hidden states of encoder.
            dropout: Dropout used for other layers except the encoder.
            span_width_feature_size: Span width feature size.
            ffnn_size: Feedforward size.
            ffnn_depth: Number of layers of feedforward MLPs.
            argument_ratio: Ratio of candidate arguments over number of tokens.
            predicate_ratio: Ratio of candidate predicates over number of tokens.
            max_arg_width: Maximum argument width.
            mlp_label_size: Feature size for label representation.
            enforce_srl_constraint: Enforce SRL constraints (number of core ARGs etc.).
            use_gold_predicates: Use gold predicates instead of predicting them.
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            use_biaffine: ``True`` to use biaffine (:cite:`dozat:17a`) instead of lineary layer for label prediction.
            loss_reduction: The loss reduction used in aggregating losses.
            with_argument: The delimiter between tokens in arguments to be used for joining tokens for outputs.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, max_seq_len=None,
                         **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, isinstance(data, list), logger, transform)
        if isinstance(data, str) and max_seq_len:
            dataset.prune(lambda x: x['token_length'] > max_seq_len, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        return SpanRankingSemanticRoleLabeler.update_metrics(self, batch, {'prediction': prediction},
                                                             tuple(metric.values()))

    def decode_output(self,
                      output: Dict[str, Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder, **kwargs) -> Union[Dict[str, Any], Any]:
        return SpanRankingSemanticRoleLabeler.decode_output(self, output, batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return output['loss']

    def compute_kd_loss(self, batch: Dict[str, Any],
                        output_student: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                        output_teacher: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                        temperature_function: TemperatureFunctionType,
                        kd_loss_function: KDLossFunctionType) -> torch.FloatTensor:
        losses = []
        # Match the scores of candidates
        arg_scores_s = output_student['candidate_arg_scores'][output_student['candidate_arg_mask']]
        arg_scores_t = output_teacher['candidate_arg_scores'][output_teacher['candidate_arg_mask']]
        # Somehow mse loss performs worse than ce loss
        losses.append(kd_mse_loss(arg_scores_s, arg_scores_t, temperature_function(arg_scores_s, arg_scores_t)))
        pred_scores_s = output_student['candidate_pred_scores'][batch['mask']]
        pred_scores_t = output_teacher['candidate_pred_scores'][batch['mask']]
        losses.append(
            kd_mse_loss(pred_scores_s, pred_scores_t, temperature_function(pred_scores_s, pred_scores_t)))
        # If topK happens to select the same args and preds as SRL, match the SRL scores
        if output_student['arg_starts'].equal(output_teacher['arg_starts']) and output_student['arg_ends'].equal(
                output_teacher['arg_ends']) and output_student['predicates'].equal(output_teacher['predicates']):
            srl_scores_s = output_student['srl_scores'][output_student['srl_mask']]
            srl_scores_t = output_teacher['srl_scores'][output_teacher['srl_mask']]
            losses.append(
                kd_loss_function(srl_scores_s, srl_scores_t, temperature_function(srl_scores_s, srl_scores_t)))
        return sum(losses)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return SpanRankingSRLDecoder(encoder_size, len(self.vocabs.srl_label), self.config)

    def build_metric(self, **kwargs):
        predicate_f1, end_to_end_f1 = SpanRankingSemanticRoleLabeler.build_metric(self, **kwargs)
        return SemanticRoleLabelingMetrics({'predicate': predicate_f1, 'e2e': end_to_end_f1})

    def build_criterion(self, **kwargs):
        pass

    def input_is_flat(self, data) -> bool:
        return SpanRankingSemanticRoleLabeler.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        return SpanRankingSemanticRoleLabeler.format_dict_to_results(batch['token'], prediction, exclusive_offset=True,
                                                                     with_predicate=True,
                                                                     with_argument=self.config.get('with_argument',
                                                                                                   ' '),
                                                                     label_first=True)
