# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-17 20:52
import logging
import os
from copy import deepcopy
from typing import Dict, Any, Union

import torch
from torch.utils.data import DataLoader

from elit.common.cache import RandomAccessFileCache
from elit.common.dataset import MultiTaskDataLoader
from elit.common.structure import ConfigTracker
from elit.components.mtl.tasks import TemperatureFunctionType, KDLossFunctionType
from elit.losses.homoscedastic_loss_weighted_sum import HomoscedasticLossWeightedSum
from elit.metrics.mtl import MetricDict
from elit.utils.time_util import CountdownTimer


class SelfTeaching(ConfigTracker):
    def __init__(self, after_epochs, loss_weighted_sum=False, cache_prefix=None) -> None:
        super().__init__({'after_epochs': after_epochs, 'loss_weighted_sum': loss_weighted_sum})
        self.cache_prefix = cache_prefix
        self.after_epochs = after_epochs
        self.trn: MultiTaskDataLoader = None
        self.best_scores = MetricDict()
        self.cache: Dict[str, RandomAccessFileCache] = dict()
        self.passed_after_epochs = False
        self.loss_weighted_sum: Union[bool, HomoscedasticLossWeightedSum] = loss_weighted_sum

    def on_trn_begin(self, trn: DataLoader, device=None):
        while not isinstance(trn, MultiTaskDataLoader) and hasattr(trn, 'dataset'):
            trn = trn.dataset
        assert isinstance(trn, MultiTaskDataLoader), \
            f'Unable to find MultiTaskDataLoader in DataLoader {type(trn)}. Make sure it has a dataset property ' \
            f'referring to nested dataloader'
        self.trn: MultiTaskDataLoader = trn
        if self.loss_weighted_sum:
            self.loss_weighted_sum = HomoscedasticLossWeightedSum(len(trn.dataloaders))
            if device:
                self.loss_weighted_sum.to(device)

    @torch.no_grad()
    def on_trn_epoch_end(self, epoch, metrics: MetricDict, mtl, logger: logging.Logger):
        """

        Args:
            epoch:
            metrics: 
            mtl (hanlp.components.mtl.multi_task_learning.MultiTaskLearning): The caller.
            logger:
        """
        if epoch >= self.after_epochs:
            self.passed_after_epochs = True
            best_scores = self.best_scores
            mtl.model.training = False
            cls_is_bos = any([x.cls_is_bos for x in mtl.tasks.values()])
            sep_is_eos = any([x.sep_is_eos for x in mtl.tasks.values()])
            for teacher_name, score in metrics.items():
                # noinspection PyUnboundLocalVariable
                if score > best_scores.get(teacher_name, -1):
                    best_scores[teacher_name] = deepcopy(score)
                    cache = self.cache[teacher_name] = RandomAccessFileCache(
                        filename=f'{self.cache_prefix}/{os.getpid()}-{teacher_name}.pkl' if self.cache_prefix else None)
                    for student_name, student in mtl.tasks.items():
                        if student_name == teacher_name:
                            continue
                        dataloader = self.trn.dataloaders[student_name]
                        timer = CountdownTimer(len(dataloader))
                        for batch in dataloader:
                            batch_key = self.batch_key(batch['token'], cls_is_bos, sep_is_eos)
                            if batch_key not in cache:
                                output_dict, _ = mtl.feed_batch(batch, teacher_name, cls_is_bos=cls_is_bos,
                                                                sep_is_eos=sep_is_eos, run_transform=True)

                                cache[batch_key] = output_dict[teacher_name]['output']
                            timer.log(f'Decoding {student_name} data for self-teaching with {teacher_name} of {score}',
                                      ratio_percentage=None, newline=False)
                        timer.erase()
            mtl.model.training = True

    @staticmethod
    def batch_key(token, cls_is_bos, sep_is_eos):
        tokens = sum([x[1 if cls_is_bos else 0:-1 if sep_is_eos else len(x)] for x in token], [])
        # return hash('\t'.join(tokens))
        return ' '.join(tokens)

    def compute_kd_loss(self,
                        major_loss,
                        student_name: str,
                        batch: Dict[str, Any],
                        cls_is_bos,
                        sep_is_eos,
                        output_dict: Dict[str, Any],
                        temperature_function: TemperatureFunctionType,
                        kd_loss_function: KDLossFunctionType,
                        mtl):
        """

        Args:
            major_loss: The major loss of student which will be summed with kd losses.
            student_name:
            batch: 
            temperature_function:
            kd_loss_function: 
            mtl (hanlp.components.mtl.multi_task_learning.MultiTaskLearning): The caller.
        """
        losses = []
        for teacher_name, task in mtl.tasks.items():
            if teacher_name == student_name:
                losses.append(major_loss)
                continue
            cache = self.cache[teacher_name]
            batch_key = self.batch_key(batch['token'], cls_is_bos, sep_is_eos)
            output_student, _batch = mtl.feed_batch(batch, teacher_name, output_dict, cls_is_bos=cls_is_bos,
                                                    sep_is_eos=sep_is_eos, run_transform=True)
            losses.append(mtl.tasks[teacher_name].compute_kd_loss(_batch, output_student[teacher_name]['output'],
                                                                  cache[batch_key], temperature_function,
                                                                  kd_loss_function))
        return self.loss_weighted_sum(*losses) if self.loss_weighted_sum else sum(losses)
