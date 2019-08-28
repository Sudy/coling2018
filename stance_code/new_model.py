import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import SequentialRepr
from utils import sequence_mask, softmax_with_mask
from baselines import BaseNetwork, AttNet
from transformer import make_model, PositionalEncoding, attention, PositionwiseFeedForward, LayerNorm
# import ipdb
# import math

class TransEncoder(BaseNetwork):

  """
    use gru_avg for sentence representation
    use tranformer encoder for context-aware document representation
  """

  def __init__(self, config):
    super(TransEncoder, self).__init__(config)

    self.config = config

    self.w2s = SequentialRepr(config, \
      input_dim = config.embed_dim, mode = "lstm")
    self.pe = PositionalEncoding(config.hidden_dim, config.dropout)

    self.s2d = make_model(N = config.num_layers,\
      d_model = config.hidden_dim, dropout = config.dropout)
    
    self.layer_norm = LayerNorm(config.hidden_dim)

    self.add_att = AttNet(config, config.hidden_dim)
    self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)
    # self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim),\
    #  nn.ReLU(), nn.Linear(config.hidden_dim, config.num_classes))

  #embed sentence with respect to the target
  def embed_ts(self, tgt_hiddens,\
   sent_hiddens, tgt_mask, sent_mask, tgt_nword):

    max_num_doc = sent_hiddens.size(0) // tgt_hiddens.size(0)
    if max_num_doc > 1:
      tgt_hiddens = tgt_hiddens.repeat(max_num_doc, 1, 1)
      tgt_mask = tgt_mask.repeat(max_num_doc, 1, 1)
      tgt_nword = tgt_nword.repeat(max_num_doc)

    matrix_score = torch.matmul(\
      tgt_hiddens, sent_hiddens.transpose(1, -1))# / math.sqrt(tgt_hiddens.size(-1))
    # ipdb.set_trace()
    matrix_mask = torch.matmul(tgt_mask, sent_mask.transpose(1, -1))

    matrix_score = matrix_score.masked_fill(matrix_mask == 0., -1e9)
    weight = torch.sum(F.softmax(matrix_score, dim = -1), dim = 1) / \
      tgt_nword.unsqueeze(-1).float()
    weight = weight / weight.sum(dim = -1, keepdim = True)

    ld_repr = torch.sum(weight.unsqueeze(-1) * sent_hiddens, dim = 1)

    return ld_repr, max_num_doc


  def embed_body(self, dcmt):

    text, ndoc, nword = dcmt

    # target, 
    max_num_word = text.size(-1)
    #(batch_size, max_num_sent, 1)
    sent_mask = sequence_mask(ndoc, device = self.config.gpu).unsqueeze(-1)
    #(batch_size * max_num_sent, max_num_word, 1)
    word_mask = sequence_mask(nword, device = self.config.gpu).view(-1, max_num_word, 1)
    #(batch_size * max_num_sent, max_num_words, embed_dim)
    text_embed = self.embed(text.view(-1, max_num_word))

    #(batch_size * max_num_sent, max_num_word, output_dim)
    sent_hiddens, _ = self.w2s(text_embed, mask = word_mask)

    return sent_hiddens, sent_mask, word_mask 

  def embed_doc(self, target, lead, dcmt):

    tgt_hiddens, _, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.w2s)
    ld_hiddens, _, __, ld_mask =\
     self.embed_sent(lead, self.w2s)

    tl_repr, _ = self.embed_ts(\
      tgt_hiddens, ld_hiddens, tgt_mask, ld_mask, tgt_nword)

    sent_hiddens, sent_mask, word_mask = self.embed_body(dcmt)

    tgt_sents, max_num_doc = self.embed_ts(tgt_hiddens,\
     sent_hiddens, tgt_mask, word_mask, tgt_nword)

    tgt_sents = tgt_sents.view(-1, max_num_doc, tgt_sents.size(-1))

    ctx_sent_repr = self.pe(tgt_sents)

    # ctx_sent_repr, _ = attention(sent_repr, sent_repr,\
    #  sent_repr, mask = sent_mask, dropout = self.dropout)

    ctx_sent_repr = self.s2d(ctx_sent_repr, sent_mask)

    output, weights = self.add_att(query = tl_repr,\
     facts = ctx_sent_repr, fact_mask = sent_mask)

    return torch.cat((tl_repr, output), dim = -1), weights



class HNEncoder(BaseNetwork):

  """
  use word-word alignment for target dependent abstract learning 
  use hierarchical for document representation

  """

  def __init__(self, config):
    super(HNEncoder, self).__init__(config)

    self.config = config

    self.w2s_tl = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm")
    self.w2s = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm_att")
    self.s2d = SequentialRepr(config, input_dim = config.hidden_dim, mode = "lstm")

    self.add_att = AttNet(config, config.hidden_dim)
    self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

  # embed target and lead sentence
  def embed_tl(self, target, lead):

    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.w2s_tl)

    ld_hiddens, ld_output, ld_nword, ld_mask =\
     self.embed_sent(lead, self.w2s_tl)

    matrix_score = torch.matmul(tgt_hiddens, ld_hiddens.transpose(1, -1))
    matrix_mask = torch.matmul(tgt_mask, ld_mask.transpose(1, -1))

    matrix_score = matrix_score.masked_fill(matrix_mask == 0., -1e9)
    weight = torch.sum(F.softmax(matrix_score, dim = -1), dim = 1) / \
      tgt_nword.unsqueeze(-1).float()
    weight = weight / weight.sum(dim = -1, keepdim = True)

    ld_repr = torch.sum(weight.unsqueeze(-1) * ld_hiddens, dim = 1)

    return ld_repr, tgt_output


  def embed_body(self, tgt_output, dcmt):

    #(batch_size, max_num_sent, max_num_word)
    text, ndoc, nword = dcmt

    # target, 
    max_num_word = text.size(-1)
    #(batch_size, max_num_sent, 1)
    sent_mask =  sequence_mask(ndoc, device = self.config.gpu).unsqueeze(-1)
    #(batch_size * max_num_sent, max_num_word, 1)
    word_mask = sequence_mask(nword, device = self.config.gpu).view(-1, max_num_word, 1)

    #(batch_size * max_num_sent, max_num_words, embed_dim)
    text_embed = self.embed(text.view(-1, max_num_word))

    #(batch_size * max_num_sent, output_dim)
    _, sent_repr = self.w2s(text_embed, mask = word_mask, init = tgt_output[1])

    sent_repr = self.dropout(sent_repr)
    sent_repr = sent_repr.view(*text.size()[:2], -1)
    ctx_sent_repr, _ = self.s2d(sent_repr, mask = sent_mask)
    ctx_sent_repr = self.dropout(ctx_sent_repr)

    return ctx_sent_repr, sent_mask


  def embed_doc(self, target, lead, dcmt):

    tl_repr, tgt_output = self.embed_tl(target, lead)
    ctx_sent_repr, sent_mask = self.embed_body(tgt_output, dcmt)

    output, weight = self.add_att(query = tl_repr,\
     facts = ctx_sent_repr, fact_mask = sent_mask)

    return torch.cat((tl_repr, output), dim = -1), weight


class TCEncoder(BaseNetwork):

  """use conditional encoding for target-abstract representation
   and transformer encoder for target-dependent representation learning"""
  def __init__(self, config):
    super(TCEncoder, self).__init__(config)
    self.config = config

    self.t2v = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm")
    self.w2s = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm_att")

    self.pe = PositionalEncoding(config.hidden_dim, config.dropout)
    self.s2d = make_model(N = config.num_layers,\
      d_model = config.hidden_dim, dropout = config.dropout)
    
    self.add_att = AttNet(config, config.hidden_dim)

    self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)


  #embed sentence with respect to the target
  def embed_ts(self, tgt_hiddens,\
   sent_hiddens, tgt_mask, sent_mask, tgt_nword):

    max_num_doc = sent_hiddens.size(0) // tgt_hiddens.size(0)

    if max_num_doc > 1:
      tgt_hiddens = tgt_hiddens.repeat(max_num_doc, 1, 1)
      tgt_mask = tgt_mask.repeat(max_num_doc, 1, 1)
      tgt_nword = tgt_nword.repeat(max_num_doc)

    matrix_score = torch.matmul(\
      tgt_hiddens, sent_hiddens.transpose(1, -1))# / math.sqrt(tgt_hiddens.size(-1))
    # ipdb.set_trace()
    matrix_mask = torch.matmul(tgt_mask, sent_mask.transpose(1, -1))

    matrix_score = matrix_score.masked_fill(matrix_mask == 0., -1e9)
    weight = torch.sum(F.softmax(matrix_score, dim = -1), dim = 1) / \
      tgt_nword.unsqueeze(-1).float()
    weight = weight / weight.sum(dim = -1, keepdim = True)

    ld_repr = torch.sum(weight.unsqueeze(-1) * sent_hiddens, dim = 1)

    return ld_repr, max_num_doc


  def embed_body(self, dcmt):

    text, ndoc, nword = dcmt
    # target, 
    max_num_word = text.size(-1)
    #(batch_size, max_num_sent, 1)
    sent_mask = sequence_mask(ndoc, device = self.config.gpu).unsqueeze(-1)
    #(batch_size * max_num_sent, max_num_word, 1)
    word_mask = sequence_mask(nword, device = self.config.gpu).view(-1, max_num_word, 1)
    #(batch_size * max_num_sent, max_num_words, embed_dim)
    text_embed = self.embed(text.view(-1, max_num_word))

    #(batch_size * max_num_sent, max_num_word, output_dim)
    sent_hiddens, _ = self.w2s(text_embed, mask = word_mask)

    return sent_hiddens, sent_mask, word_mask


  def embed_doc(self, target, lead, dcmt):

    #conditional encoding
    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.t2v)

    ld_hiddens, ld_output, ld_nword, ld_mask =\
     self.embed_sent(lead, self.w2s, h0 = tgt_output[1])


    sent_hiddens, sent_mask, word_mask = self.embed_body(dcmt)

    tgt_sents, max_num_doc = self.embed_ts(tgt_hiddens,\
     sent_hiddens, tgt_mask, word_mask, tgt_nword)

    tgt_sents = tgt_sents.view(-1, max_num_doc, tgt_sents.size(-1))

    ctx_sent_repr = self.pe(tgt_sents)

    # ctx_sent_repr, _ = attention(sent_repr, sent_repr,\
    #  sent_repr, mask = sent_mask, dropout = self.dropout)

    ctx_sent_repr = self.s2d(ctx_sent_repr, sent_mask)

    output, weights = self.add_att(query = ld_output,\
     facts = ctx_sent_repr, fact_mask = sent_mask)

    return torch.cat((ld_output, output), dim = -1), weights
