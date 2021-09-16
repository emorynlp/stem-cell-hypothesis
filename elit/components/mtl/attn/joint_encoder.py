# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-02 13:32
from typing import Optional, Union, Dict, Any

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from elit.components.mtl.attn.attn import TaskAttention
from elit.components.mtl.attn.transformer import JointEncoder
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule, ContextualWordEmbedding
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.layers.transformers.utils import pick_tensor_for_each_token


class JointContextualWordEmbeddingModule(ContextualWordEmbeddingModule):

    def __init__(self, field: str, transformer: str, transformer_tokenizer: PreTrainedTokenizer, average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None, word_dropout=None,
                 max_sequence_length=None, ret_raw_hidden_states=False, transformer_args: Dict[str, Any] = None,
                 trainable=True, training=True) -> None:
        super().__init__(field, transformer, transformer_tokenizer, average_subwords, scalar_mix, word_dropout,
                         max_sequence_length, ret_raw_hidden_states, transformer_args, trainable, training)
        self.adapter: TaskAttention = None

    def forward(self, batch: dict, mask=None, **kwargs):
        input_ids: torch.LongTensor = batch[f'{self.field}_input_ids']
        if self.max_sequence_length and input_ids.size(-1) > self.max_sequence_length:
            raise NotImplementedError('Sentence length exceeded and sliding window has not been implemented yet')
        token_span: torch.LongTensor = batch.get(f'{self.field}_token_span', None)
        token_type_ids: torch.LongTensor = batch.get(f'{self.field}_token_type_ids', None)
        attention_mask = input_ids.ne(0)
        if self.word_dropout:
            input_ids = self.word_dropout(input_ids)

        # noinspection PyTypeChecker
        transformer: JointEncoder = self.transformer
        encoder_outputs = transformer(input_ids, attention_mask, token_type_ids)
        outputs = dict()
        for task_name, encoder_output in encoder_outputs.items():
            encoder_output = encoder_output[0]
            outputs[task_name] = pick_tensor_for_each_token(encoder_output, token_span, self.average_subwords)
        return outputs


class JointContextualWordEmbedding(ContextualWordEmbedding):

    def module(self, training=True, **kwargs) -> Optional[nn.Module]:
        return JointContextualWordEmbeddingModule(self.field,
                                                  self.transformer,
                                                  self._transformer_tokenizer,
                                                  self.average_subwords,
                                                  self.scalar_mix,
                                                  self.word_dropout,
                                                  self.max_sequence_length,
                                                  self.ret_raw_hidden_states,
                                                  self.transformer_args,
                                                  self.trainable,
                                                  training=training)
