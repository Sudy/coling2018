import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseRepr, SequentialRepr
from utils import sequence_mask, softmax_with_mask


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
    # norm_weights = F.dropout(norm_weights, p = self.config.dropout, training = self.training)
    return torch.sum(norm_weights * facts, dim = 1), norm_weights

class BaseNetwork(BaseRepr):
  def __init__(self, config):
    super(BaseNetwork, self).__init__(config)
    self.config = config

    # self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim),\
    #  nn.Tanh(), nn.Dropout(p = config.dropout),\
    #   nn.Linear(config.hidden_dim, config.num_classes))    
    self.fc = nn.Linear(config.hidden_dim, config.num_classes)
    self.dropout = nn.Dropout(p = config.dropout)

  def embed_sent(self, sent, encoder, h0 = None):
    snt, stn_nword = sent
    word_mask = sequence_mask(stn_nword, device = self.config.gpu).unsqueeze(-1)
    snt_embed = self.embed(snt)
    hiddens, output = encoder(snt_embed, init = h0, mask = word_mask)
    return hiddens, output, stn_nword, word_mask

  def embed_doc(self, target, lead, dcmt):
    raise Exception("Not implemented!")

  def forward(self, batch):
    doc_repr, weights = self.embed_doc(batch.claim, batch.abst, batch.body)
    pred = self.fc(doc_repr)
    return F.log_softmax(pred, dim = -1), weights

# use claim and headline information for classifying the document
class HLNetwork(BaseNetwork):

  """docstring for HLNetwork"""
  def __init__(self, config):
    super(HLNetwork, self).__init__(config)

    self.w2s = SequentialRepr(config, \
      input_dim = config.embed_dim, mode = "lstm")

  def embed_doc(self, target, lead, dcmt):

    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.w2s)
    ld_hiddens, ld_output, ld_nword, ld_mask =\
     self.embed_sent(lead, self.w2s)

    matrix_score = torch.matmul(tgt_hiddens, ld_hiddens.transpose(1, -1))
    matrix_mask = torch.matmul(tgt_mask, ld_mask.transpose(1, -1))

    matrix_score = matrix_score.masked_fill(matrix_mask == 0., -1e9)

    weight = torch.sum(F.softmax(matrix_score, dim = -1), dim = 1) / \
      tgt_nword.unsqueeze(-1).float()
    weight = weight / weight.sum(dim = -1, keepdim = True)
    # weight = self.dropout(weight)

    ld_repr = torch.sum(weight.unsqueeze(-1) * ld_hiddens, dim = 1)
    return ld_repr, None

class ConditionEncode(BaseNetwork):
  def __init__(self, config):
    super(ConditionEncode, self).__init__(config)
    self.w2s = SequentialRepr(config,\
     input_dim = config.embed_dim, mode = "lstm")

  def embed_doc(self, target, lead, dcmt):

    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.w2s)

    ld_hiddens, ld_output, ld_nword, ld_mask =\
     self.embed_sent(lead, self.w2s, h0 = tgt_output[1])
    output = torch.sum(ld_hiddens * ld_mask, dim = 1) / ld_nword.unsqueeze(-1).float()

    return self.dropout(output), None


#hard attention for document representation
class HNetwork(BaseNetwork):

  """docstring for HNetwork"""
  def __init__(self, config):
    super(HNetwork, self).__init__(config)

    sent_mode = doc_mode = None

    if config.model == "HN":
      # hierarchical average network
      sent_mode = doc_mode = "lstm_avg"

    elif config.model == "HAN":
      # hierarchical attention network
      sent_mode = doc_mode = "lstm_att"

    else: raise Exception("mode %s not defined" % config.model)

    self.t2v = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm")
    self.w2s = SequentialRepr(config, input_dim = config.embed_dim, mode = sent_mode)
    self.s2d = SequentialRepr(config, input_dim = config.hidden_dim, mode = doc_mode)

    self.dropout = nn.Dropout(p = config.dropout)
    self.fc = nn.Linear(config.hidden_dim, config.num_classes)


  def embed_doc(self, target, lead, dcmt):

    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.t2v)

    #(batch_size, max_num_sent, max_num_word)
    text, ndoc, nword = dcmt

    # target, 
    max_num_word = text.size(-1)
    #(batch_size, max_num_sent, 1)
    doc_mask =  sequence_mask(ndoc, device = self.config.gpu).unsqueeze(-1)
    #(batch_size * max_num_sent, max_num_word, 1)
    word_mask = sequence_mask(nword, device = self.config.gpu).view(-1, max_num_word, 1)

    #(batch_size * max_num_sent, max_num_words, embed_dim)
    text_embed = self.embed(text.view(-1, max_num_word))

    #(batch_size * max_num_sent, output_dim)
    _, sent_repr = self.w2s(text_embed, mask = word_mask, init = tgt_output[1])

    sent_repr = self.dropout(sent_repr)
    sent_repr = sent_repr.view(*text.size()[:2], -1)

    _, doc_repr = self.s2d(sent_repr, mask = doc_mask, init = tgt_output[1])
    doc_repr = self.dropout(doc_repr)

    return doc_repr, _



#hard attention for document representation
class TargetedHNetwork(BaseNetwork):

  """docstring for HNetwork"""
  def __init__(self, config):
    super(TargetedHNetwork, self).__init__(config)

    sent_mode = doc_mode = None


    self.t2v = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm")
    self.w2s = SequentialRepr(config, input_dim = config.embed_dim, mode = "lstm_att")
    self.s2d = SequentialRepr(config, input_dim = config.hidden_dim, mode = "lstm")

    self.add_att = AttNet(config, config.hidden_dim)
    self.dropout = nn.Dropout(p = config.dropout)
    self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

  def embed_tl(self, target, lead):

    tgt_hiddens, tgt_output, tgt_nword, tgt_mask =\
     self.embed_sent(target, self.t2v)

    ld_hiddens, ld_output, ld_nword, ld_mask =\
     self.embed_sent(lead, self.w2s, h0 = tgt_output[1])
    # output = torch.sum(ld_hiddens * ld_mask, dim = 1) / ld_nword.unsqueeze(-1).float()

    # ld_output = self.dropout(ld_output)

    return ld_output, tgt_output

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
