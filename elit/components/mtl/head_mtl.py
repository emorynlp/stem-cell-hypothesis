# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-22 10:54
import os
import torch
from typing import Union

from transformers.models.deberta.modeling_deberta import DebertaLayer, DebertaAttention
from transformers.models.electra.modeling_electra import ElectraAttention
from transformers.models.roberta.modeling_roberta import RobertaAttention

from elit.layers.transformers.pt_imports import BertModel
from transformers import BertLayer
from transformers.models.bert.modeling_bert import BertAttention
from elit.common.cache import SequentialFileCache
from elit.components.mtl.multi_task_learning import MultiTaskLearning
from elit.layers.transformers.pt_imports import AutoModel


def sum_span(h, token_span, mean):
    if token_span.size(-1) > 1:
        batch_size = h.size(0)
        h_span = h.gather(1, token_span.view(batch_size, -1).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        h_span = h_span.view(batch_size, *token_span.shape[1:], -1)
        n_sub_tokens = token_span.ne(0)
        n_sub_tokens[:, 0, 0] = True
        h_span = (h_span * n_sub_tokens.unsqueeze(-1)).sum(2)
        if mean:
            n_sub_tokens = n_sub_tokens.sum(-1).unsqueeze(-1)
            zero_mask = n_sub_tokens == 0
            if torch.any(zero_mask):
                n_sub_tokens[zero_mask] = 1  # avoid dividing by zero
            embed = h_span / n_sub_tokens
        else:
            embed = h_span
    else:
        embed = h.gather(1, token_span[:, :, 0].unsqueeze(-1).expand(-1, -1, h.size(-1)))
    return embed


class BertAttentionValue(BertAttention):

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[:1]  # output attended value
        return outputs


class RobertaAttentionValue(RobertaAttention):

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[:1]  # output attended value
        return outputs


class ElectraAttentionValue(ElectraAttention):

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[:1]  # output attended value
        return outputs


class DebertaAttentionValue(DebertaAttention):

    def forward(
            self,
            hidden_states,
            attention_mask,
            return_att=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, (att_matrix, self_output))  # output attended value
        else:
            return attention_output


class HeadMultiTaskLearning(MultiTaskLearning):
    def dump_attention_per_head(self, save_dir, subfoler=''):
        if subfoler:
            path = f'{save_dir}/{subfoler}/attn.pt'
        else:
            path = f'{save_dir}/attn.pt'
        if os.path.isfile(path):
            return SequentialFileCache(None, None, path, delete=False, device=-1)
        logger = self.build_logger('head', save_dir)
        dataloader = self.build_dataloader('tst', self.config['batch_size'], device=self.device, logger=logger)
        transformer = self.model.encoder.transformer
        if subfoler == 'static':
            transformer = AutoModel.from_pretrained(self.config['encoder'].transformer).to(self.device)
        H = self.model.encoder.transformer.config.num_attention_heads
        transformer.eval()

        @torch.no_grad()
        def gen():
            for task_name, batch in dataloader:
                input_ids: torch.LongTensor = batch['token_input_ids']
                token_span: torch.LongTensor = batch.get('token_token_span', None)
                token_type_ids: torch.LongTensor = batch.get('token_token_type_ids', None)
                attention_mask = input_ids.ne(0)
                outputs = transformer(input_ids, attention_mask, token_type_ids, output_attentions=True)
                attentions = outputs.attentions
                B, L, S = token_span.size()
                token_span_3d = token_span.unsqueeze(1).repeat(1, H, 1, 1)  # (B, H, L, S)
                token_span_flat = token_span_3d.flatten(0, 1)
                pooled_attentions = []
                for each in attentions:
                    pooled = sum_span(each.flatten(0, 1), token_span_flat, True)
                    pooled = sum_span(pooled.transpose(1, 2), token_span_flat, False)
                    pooled = pooled.transpose(1, 2).view(B, H, L, L)
                    pooled_attentions.append(pooled)
                yield batch, pooled_attentions

        cache = SequentialFileCache(gen(), len(dataloader), path, delete=False, device=-1)
        return cache

    def dump_attended_values_per_head(self, save_dir, subfoler=''):
        if subfoler:
            path = f'{save_dir}/{subfoler}/v.pt'
        else:
            path = f'{save_dir}/v.pt'
        if os.path.isfile(path):
            return SequentialFileCache(None, None, path, delete=False, device=-1)
        logger = self.build_logger('head', save_dir)
        dataloader = self.build_dataloader('tst', self.config['batch_size'], device=self.device, logger=logger)
        # noinspection PyTypeChecker
        transformer: BertModel = self.model.encoder.transformer
        if subfoler == 'static':
            transformer = AutoModel.from_pretrained(self.config['encoder'].transformer)
        for layer in transformer.encoder.layer:
            layer: Union[BertLayer, DebertaLayer] = layer
            if isinstance(layer.attention, BertAttention):
                # noinspection PyTypeChecker
                attn = BertAttentionValue(transformer.config)
                attn.load_state_dict(layer.attention.state_dict(), strict=True)
                layer.attention = attn
            elif isinstance(layer.attention, RobertaAttention):
                # noinspection PyTypeChecker
                attn = RobertaAttentionValue(transformer.config)
                attn.load_state_dict(layer.attention.state_dict(), strict=True)
                layer.attention.self = attn
            elif isinstance(layer.attention, ElectraAttention):
                # noinspection PyTypeChecker
                attn = ElectraAttentionValue(transformer.config)
                attn.load_state_dict(layer.attention.state_dict(), strict=True)
                layer.attention.self = attn
            elif isinstance(layer.attention, DebertaAttention):
                # noinspection PyTypeChecker
                attn = DebertaAttentionValue(transformer.config)
                attn.load_state_dict(layer.attention.state_dict(), strict=True)
                layer.attention.self = attn
            else:
                raise NotImplementedError('Unsupported transformer')

        transformer = transformer.to(self.device)
        transformer.eval()
        H = self.model.encoder.transformer.config.num_attention_heads

        @torch.no_grad()
        def gen():
            for task_name, batch in dataloader:
                input_ids: torch.LongTensor = batch['token_input_ids']
                token_span: torch.LongTensor = batch.get('token_token_span', None)
                token_type_ids: torch.LongTensor = batch.get('token_token_type_ids', None)
                attention_mask = input_ids.ne(0)
                outputs = transformer(input_ids, attention_mask, token_type_ids, output_attentions=True)
                attentions = outputs.attentions
                B, L, S = token_span.size()
                pooled_attentions = []
                for each in attentions:
                    if isinstance(each, tuple):
                        each = each[-1]
                    pooled = sum_span(each, token_span, True)
                    pooled = pooled.view(B, L, H, -1)
                    pooled_attentions.append(pooled)
                yield batch, pooled_attentions

        cache = SequentialFileCache(gen(), len(dataloader), path, delete=False, device=-1)
        return cache
