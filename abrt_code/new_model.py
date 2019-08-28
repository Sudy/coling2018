import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines import SequentialRepr, AttNet, BaseHNetwork
from utils import sequence_mask
from transformer import make_model, PositionalEncoding, attention

class TransEncoder(BaseHNetwork):

  """
    use lstm for sentence representation
    use tranformer encoder for context-aware document representation
  """

  def __init__(self, config):
    super(TransEncoder, self).__init__(config)

    self.sent_repr_dim = config.hidden_dim
    self.w2s = SequentialRepr(config, \
      input_dim = config.embed_dim, mode = "lstm")
    # self.w2s_tl = SequentialRepr(config,\
      # input_dim = config.embed_dim, mode = "lstm")
    
    # self.s2d = SequentialRepr(config,
    #  input_dim = config.hidden_dim, mode = "lstm")

    self.pe = PositionalEncoding(self.sent_repr_dim, config.dropout)  
    self.s2d = make_model(N = config.num_layers,\
      d_model = self.sent_repr_dim, dropout = config.dropout)

    self.satt_layer = AttNet(config, config.hidden_dim)
    self.datt_layer = AttNet(config, config.hidden_dim * 2)

    self.dropout = nn.Dropout(p = config.dropout)
    self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)


  def embed_lead(self, leads):

    leads, nleads, lead_nwords = leads
    max_num_lead = lead_nwords.size(-1)

    # (batch_size, max_num_doc)
    lead_mask = sequence_mask(nleads, device = self.config.gpu)
    # (batch_size, max_num_doc, max_seqlen)
    lead_word_mask = sequence_mask(lead_nwords, device = self.config.gpu)
    lead_word_mask = lead_word_mask.view(-1, lead_word_mask.size(-1), 1)

    leads = leads.view(-1, leads.size(-1))
    #(batch_size * max_num_doc, max_seqlen, embed_dim)
    leads_embeded = self.wembed(leads)

    #(batch_size * max_num_doc, max_seqlen, hidden_dim)
    lead_hiddens, _ = self.w2s(leads_embeded, mask = lead_word_mask)
    return lead_hiddens, lead_word_mask, max_num_lead


  # embed target and lead sentence
  def embed_tl(self, target, lead):

    # ipdb.set_trace()
    # (batch_size, embed_dim)
    target_embeded = self.embed_target(target)
    # target_embeded = self.wembed(target)

    #(batch_size)
    lead_hiddens, lead_word_mask, max_num_lead = self.embed_lead(lead)
    # ipdb.set_trace()
    target_embeded = target_embeded.unsqueeze(1).repeat(max_num_lead, 1, 1)

    tl_reprs, p_attn = attention(target_embeded, lead_hiddens, lead_hiddens,\
     mask = lead_word_mask.transpose(-1, -2), dropout = self.dropout,  scale = True)

    return tl_reprs, target_embeded, max_num_lead

  def embed_body(self, dcmt, target_embeded):
    
    dcmts, ndocs, nsents, dcmt_nwords = dcmt

    doc_mask = sequence_mask(ndocs, device = self.config.gpu)
    doc_sent_mask = sequence_mask(nsents, device = self.config.gpu)
    doc_word_mask = sequence_mask(dcmt_nwords, device = self.config.gpu)

    batch_size, max_num_doc, max_num_sent, max_seqlen = dcmts.size()
    dcmts_reshaped = dcmts.view(-1, max_seqlen)

    #(batch_size * max_num_doc * max_num_sent, max_seqlen, embed_dim)
    dcmts_embeded = self.wembed(dcmts_reshaped)

    # _, sent_reprs = self.w2s(dcmts_embeded, \
    #   mask = doc_word_mask.view(-1, doc_word_mask.size(-1), 1))

    #(batch_size * max_num_doc * max_num_sent, max_seqlen, hidden_dim * 2)
    
    hiddens, _ = self.w2s(dcmts_embeded)
    # ipdb.set_trace()
    target_embeded_expand = target_embeded.repeat(max_num_sent, 1, 1)
    sent_reprs, _ = attention(target_embeded_expand, hiddens, hiddens,\
     mask = doc_word_mask.view(-1, 1, doc_word_mask.size(-1)), dropout = self.dropout, scale = True)

    #(batch_size * max_num_doc, max_num_sent, hidden_dim * 2)
    sent_reprs = sent_reprs.view(-1, max_num_sent, sent_reprs.size(-1))
    doc_sent_mask = doc_sent_mask.view(-1, doc_sent_mask.size(-1), 1)
    return sent_reprs, doc_sent_mask, doc_mask

  def embed_doc(self, target, lead, dcmt):

    tl_reprs, target_embeded, max_num_doc = self.embed_tl(target, lead)
    sent_reprs, sent_mask, doc_mask = self.embed_body(dcmt, target_embeded)

    sent_reprs = self.pe(sent_reprs)
    ctx_sent_repr = self.s2d(sent_reprs, sent_mask)

    # ctx_sent_repr, _ = self.s2d(sent_reprs, mask = sent_mask)
    doc_reprs = self.satt_layer(ctx_sent_repr, tl_reprs.squeeze(1), fact_mask = sent_mask)
    
    # doc_reprs, _ = attention(tl_reprs,\
     # ctx_sent_repr, ctx_sent_repr, mask =  sent_mask.transpose(-1,-2), scale = False)

    # doc_reprs = self.add_att(tl_reprs, ctx_sent_repr, ctx_sent_repr,\
    #   mask = sent_mask)

    tl_reprs = tl_reprs.view(-1, max_num_doc, tl_reprs.size(-1))
    doc_reprs = doc_reprs.view(-1, max_num_doc, doc_reprs.size(-1))

    doc_reprs = torch.cat([tl_reprs, doc_reprs], dim = -1)
    doc_reprs = self.datt_layer(facts = doc_reprs, fact_mask = doc_mask.unsqueeze(-1))

    return doc_reprs