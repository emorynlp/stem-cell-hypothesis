# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-01 11:35
import functools
from typing import Tuple

import torch

from elit.common.transform import TransformList
from elit.components.mtl.multi_task_learning import MultiTaskLearning
from elit.components.mtl.tasks import Task
from elit.transform.transformer_tokenizer import TransformerSequenceTokenizer


def create_indicator_tokens(sample: dict, task_indicator: dict):
    input_ids: list = sample['token_input_ids']
    sample['token_token_type_ids'] = [0] * len(input_ids) + [1] * len(task_indicator)
    for task, indicator in task_indicator.items():
        assert indicator not in input_ids
        sample[f'{task}_indicator_offset'] = len(input_ids)
        input_ids.append(indicator)
    return sample


class IndicatorMultiTaskLearning(MultiTaskLearning):
    def build_transform(self, task: Task, training=False) -> Tuple[TransformerSequenceTokenizer, TransformList]:
        encoder_transform, transform = super().build_transform(task, training)
        if 'task_indicator' not in self.config:
            unused_tokens = [f'[unused{i}]' for i in range(1, 100)]
            tokenizer = encoder_transform.tokenizer
            ids = tokenizer.convert_tokens_to_ids(unused_tokens)
            unused_tokens = dict(
                (x, ids[i]) for i, x in enumerate(unused_tokens) if ids[i] != tokenizer.unk_token_id)
            self.config.task_indicator = dict(zip(self.tasks, unused_tokens.values()))
        # right after tokenizer
        transform.insert(transform.index(encoder_transform) + 1,
                         functools.partial(create_indicator_tokens, task_indicator=self.config.task_indicator))
        return encoder_transform, transform

    def _encode(self, batch, task_name, output_dict=None, cls_is_bos=False, sep_is_eos=False):
        h, output_dict = super()._encode(batch, task_name, output_dict, cls_is_bos, sep_is_eos)
        indicator_offset: torch.Tensor = batch[f'{task_name}_indicator_offset']
        indicator_offset = indicator_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(-1))
        indicator_embed = output_dict['raw_hidden'].gather(1, indicator_offset)
        h *= torch.sigmoid(indicator_embed)
        return h, output_dict
