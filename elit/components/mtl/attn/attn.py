# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-01 21:32
import torch
import torch.nn as nn


class TaskAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)

    def forward(self, *embeddings):
        embeddings = torch.stack(embeddings, dim=2)  # B, L, T, C
        B, L, T, C = embeddings.shape
        embeddings = embeddings.view([B * L, T, C])
        embeddings = self.encoder(embeddings)
        embeddings = embeddings.view([B, L, T, C])
        return embeddings
