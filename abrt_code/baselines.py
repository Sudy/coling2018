import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.nn import LSTM
from utils import sequence_mask, softmax_with_mask#, softmax_with_mask, gumbel_softmax


class AttNet(nn.Module):
  """docstring for AttNet"""
  def __init__(self, config, input_dim):
    super(AttNet, self).__init__()
    self.config = config
    # self.affine = nn.Linear(input_dim, 1)
    self.w_q = nn.Linear(input_dim, input_dim, bias = False)
    self.w_f = nn.Linear(input_dim, input_dim, bias = True)

    self.affine = nn.Sequential(nn.Tanh(),\
     nn.Dropout(p = config.dropout), nn.Linear(input_dim, 1))

  #query(batch_size, 1, hidden_dim)
  #facts(batch_size, max_seqlen, hidden_dim)
  #fact_mask(batch_size, max_seqlen, 1)
  #return (batch_size, hidden_dim)
  def forward(self, facts, query = None, fact_mask = None):

    if query is not None:
      # max_seqlen = facts.size(1)
      # query = query.unsqueeze(1).repeat(1, max_seqlen, 1)
      # #(batch_size, max_seqlen, hidden_dim * 2)
      # output = torch.cat([query, facts], dim = -1)

      output = self.w_q(query.unsqueeze(1)) + self.w_f(facts)
    else:
      output = facts

    #(batch_size, max_seqlen, 1)
    weights = self.affine(output)
    norm_weights = softmax_with_mask(weights, mask = fact_mask, dim = 1)

    return torch.sum(norm_weights * facts, dim = 1)


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

        batch_size = text_embed.size(0) // init.size(0)
        init = c0 = init.unsqueeze(0).repeat(2, batch_size, 1)

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


class BaseHNetwork(nn.Module):
  """docstring for HNetwork"""
  def __init__(self, config):
    super(BaseHNetwork, self).__init__()

    self.config = config
    sent_mode = doc_mode = None

    if config.mode == "avg":
      # self.sent_repr_dim = len(config.kernel_sizes.split(',')) * config.num_filters
      sent_mode = doc_mode = "lstm_avg"
    elif config.mode == "att":
      sent_mode = doc_mode = "lstm_att"

    else: raise Exception("mode %s not defined" % config.mode)

    self.tembed = nn.Embedding(config.tvocab_size, config.embed_dim)
    self.wembed = nn.Embedding(config.wvocab_size, config.embed_dim)

    self.w2s = SequentialRepr(config, input_dim = config.embed_dim, mode = sent_mode)
    self.s2d = SequentialRepr(config, input_dim = config.hidden_dim, mode = doc_mode)
    self.datt_layer = AttNet(config, config.hidden_dim)

    self.dropout = nn.Dropout(p = config.dropout)
    self.fc = nn.Linear(config.hidden_dim, config.num_classes)


  def embed_target(self, target):
    #target(batch_size, seqlen)
    #target_nwords(batch_size)
    target, target_nwords = target
    #(batch_size, seqlen)
    target_word_mask = sequence_mask(target_nwords, device = self.config.gpu)

    #(batch_size, seqlen, embed_dim)
    # target_embed = self.wembed(target)
    target_embed = self.tembed(target)
    target_word_mask = target_word_mask.unsqueeze(-1)
    target_embed = torch.sum(target_embed * target_word_mask, \
      dim = 1) / torch.sum(target_word_mask, dim = 1)

    return target_embed

  def embed_doc(self, target, lead, dcmt):
    raise Exception("Not implemented!")

  def forward(self, batch):

    #texts, document length and sentence length
    #text(batch_size, max_num_sent, max_seqlen)
    doc_repr = self.embed_doc(batch.target, batch.leads, batch.docs)
    pred = self.fc(doc_repr)
    return F.log_softmax(pred, dim = -1), None

#hard attention for document representation
class HNetwork(BaseHNetwork):

  def __init__(self, config):
    super(HNetwork, self).__init__(config)

  def embed_body(self, target, dcmt):
  
    dcmts, ndocs, nsents, dcmt_nwords = dcmt

    doc_mask = sequence_mask(ndocs, device = self.config.gpu)
    doc_sent_mask = sequence_mask(nsents, device = self.config.gpu)
    doc_word_mask = sequence_mask(dcmt_nwords, device = self.config.gpu)

    batch_size, max_num_doc, max_num_sent, max_seqlen = dcmts.size()
    dcmts_reshaped = dcmts.view(-1, max_seqlen)

    #(batch_size * max_num_doc, embed_dim)
    target_embed = self.embed_target(target)

    #(batch_size * max_num_doc * max_num_sent, max_seqlen, embed_dim)
    dcmts_embeded = self.wembed(dcmts_reshaped)

    #(batch_size * max_num_doc * max_num_sent, hidden_dim * 2)
    _, sent_reprs = self.w2s(dcmts_embeded, init = target_embed,\
     mask = doc_word_mask.view(-1, doc_word_mask.size(-1), 1))
    sent_reprs = self.dropout(sent_reprs)

    #(batch_size * max_num_doc, max_num_sent, hidden_dim * 2)
    sent_reprs = sent_reprs.view(-1, max_num_sent, sent_reprs.size(-1))
    doc_sent_mask = doc_sent_mask.view(-1, doc_sent_mask.size(-1), 1)

    #(batch_size * max_num_doc, max_num_sent, hidden_dim * 2)
    _, doc_reprs = self.s2d(sent_reprs, mask = doc_sent_mask)

    doc_reprs = doc_reprs.view(batch_size, max_num_doc, -1)
    doc_reprs = self.dropout(doc_reprs)

    return doc_reprs, doc_mask

  def embed_doc(self, target, lead, dcmt):

    doc_reprs, doc_mask = self.embed_body(target, dcmt)
    doc_reprs = self.datt_layer(facts = doc_reprs,\
     fact_mask = doc_mask.unsqueeze(-1))
    return doc_reprs