# -*- coding:utf-8 -*-
from elit.layers.transformers.pt_imports import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
idx = tokenizer.encode(':', add_special_tokens=False)[0]
transformer = AutoModel.from_pretrained('roberta-base')
cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
embed = transformer.embeddings.word_embeddings.weight
vocab = dict((v, k) for k, v in tokenizer.get_vocab().items())

sim = cos(embed[idx].expand_as(embed), embed)
closest = sim.argmax(0).item()
_, topk = sim.topk(10)

for i in topk.tolist():
    print(vocab[i])
