import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU
from utils import sequence_mask, softmax_with_mask
# from utils import sequence_mask, softmax_with_mask, gumbel_softmax

class  CNNText(nn.Module):

  def __init__(self, args, kernel_sizes):
    super(CNNText, self).__init__()

    self.convs = nn.ModuleList([nn.Conv2d(1, args.num_filters,\
     (k, args.embed_dim)) for k in kernel_sizes])
    self.dropout = nn.Dropout(args.dropout)

  #x(batch_size, seqlen, embed_dim)
  def forward(self, text_inputs):
    #(batch_size, 1, seqlen, embed_dim)
    text_inputs = text_inputs.unsqueeze(1)
    #[(batch_size, num_filters, seqlen - filter_width)] * len(kernel_sizes)
    conv_out = [F.relu(conv(text_inputs)).squeeze(-1) for conv in self.convs]
    #[batch_size, num_filters] * len(kernel_sizes)
    pool_out = [F.max_pool1d(cv, cv.size(-1)).squeeze(-1) for cv in conv_out]
    cat_out = torch.cat(pool_out, -1)

    return cat_out

class Attention(nn.Module):

  def __init__(self, config, input_dim):
    super(Attention, self).__init__()

    self.att = nn.Sequential(\
      nn.Linear(input_dim, input_dim), nn.Tanh(),\
      nn.Linear(input_dim, 1, bias = False))
    self.dropout = nn.Dropout(p = config.dropout)

  #facts(batch_size, max_seq_len, input_dim)
  #fact_mask(batch_size, max_seq_len, 1)
  #return(batch_size, embed_dim)
  def forward(self, facts, fact_mask = None, hard = False, tau = 1, dim = -1):

    #(batch_size, max_seq_len, 1)
    scores = self.att(facts)

    if hard:
      scores = scores.squeeze(-1)
      if fact_mask is not None:
        p_attn = gumbel_softmax(scores, fact_mask.squeeze(-1), tau = tau)
      else:
        p_attn = gumbel_softmax(scores, None, tau = tau)
      return torch.sum(p_attn.unsqueeze(-1) * facts, dim = dim)

    else:

      p_attn = softmax_with_mask(scores, fact_mask, dim = dim)
      p_attn = self.dropout(p_attn)
      return torch.sum(p_attn * facts, dim = dim)

class BaseRepr(nn.Module):

  def __init__(self, config):
    super(BaseRepr, self).__init__()
    self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

#sequential representation
class SequentialRepr(nn.Module):
  def __init__(self, config, input_dim = 100, mode = 'cnn'):
    super(SequentialRepr, self).__init__()

    self.mode = mode
    self.config = config
  
    if mode == 'cnn':
      ks = [int(c) for c in config.kernel_sizes.split(',')]
      self.encoder = CNNText(config, ks)

    elif mode.startswith("gru"):
      self.encoder = nn.GRU(input_size = input_dim, batch_first = True, \
        hidden_size = config.hidden_dim,  bidirectional = True, \
        num_layers = 1)

    elif mode.startswith('lstm'):
      self.encoder = nn.LSTM(input_size = input_dim, batch_first = True, \
        hidden_size = config.hidden_dim,  bidirectional = True, \
        num_layers = 1)

    if mode.endswith('att'):
      self.att = Attention(config, config.hidden_dim)


  # text_embed(batch_size, seqlen, embed_dim)
  # mask(batch_size, seqlen, 1)
  def forward(self, text_embed, init = None, mask = None):

    hiddens, output = None, None
    if self.mode.startswith("gru") or self.mode.startswith('lstm'):

      if init is not None:

        batch_size = text_embed.size(0) // init.size(1)
        if batch_size > 1:
          c0 = init.repeat(1, batch_size, 1)
        else:
          c0 = init

        if self.mode.startswith("lstm"):
          h0 = torch.zeros_like(c0)
          init = (h0, c0)

      hiddens, output = self.encoder(text_embed, init)
      hsize = hiddens.size()
      hiddens = hiddens.view(*hsize[:-1], 2, -1)
      hiddens = hiddens.mean(dim = -2, keepdim = False)

      if self.mode.endswith('avg'):
        if mask is not None:
          output = torch.sum(hiddens * mask, dim = 1) / (torch.sum(mask, dim = 1) + 1e-8)
        else:
          output = torch.mean(hiddens, dim = 1, keepdim = False)
      elif self.mode.endswith('att'):
        output = self.att(hiddens, mask, dim = 1)

    elif self.mode == 'cnn':
      output = self.encoder(text_embed)

    else:
      raise Exception('mode %s not defined!' % self.mode)
    return hiddens, output


