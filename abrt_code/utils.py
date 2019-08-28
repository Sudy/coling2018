import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchtext.vocab import GloVe, Vectors
from ignite.engine.engine import Engine


def sequence_mask(lengths, max_len = None, device = 0):

  if not max_len:
  	max_len = torch.max(lengths)
  #(1, max_len)
  range_matrix = torch.arange(0, max_len, 1, \
    dtype = torch.float, device = device).unsqueeze(0)
  #(batch_size, 1)
  lengths_expanded = lengths.float().unsqueeze(-1)
  return torch.lt(range_matrix, lengths_expanded).float()

def softmax_with_mask(score, mask = None, dim = -1):
	if mask is not None:
		score = score.masked_fill(mask == 0., -1e9)
	p_attn = F.softmax(score, dim = dim)
	return p_attn


class Bottle(nn.Module):

	def forward(self, input):
		if len(input.size()) <= 2:
			return super(Bottle, self).forward(input)

		size = input.size()[:2]
		out = super(Bottle, self).forward(input.contiguous().view(size[0] * size[1], -1))
		return out.view(size[0], size[1], -1)

class BatchNorm(Bottle, nn.BatchNorm1d):
	pass


class MyVectors(Vectors):
	def __init__(self, **kwargs):
		super(MyVectors, self).__init__(**kwargs)
	
	def __getitem__(self, token):
		if token in self.stoi:
			return self.vectors[self.stoi[token]]
		elif '_' in token:
			num_vectors = 0
			vector = torch.Tensor(1, self.dim).zero_()
			words = token.split('_')
			for w in words:
				if w in self.stoi:
					vector += self.vectors[self.stoi[w]]
					num_vectors += 1
			if num_vectors > 0:
				vector /= num_vectors
			else: vector = self.unk_init(vector)
			
			return vector

		else:
			return self.unk_init(torch.Tensor(self.dim))