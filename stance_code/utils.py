import torch
import torch.nn.functional as F
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