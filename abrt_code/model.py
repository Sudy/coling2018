import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from baselines import BaseHNetwork, AttNet

from utils import *
from transformer import LayerNorm

class TargetedHNetwork(BaseHNetwork):
	"""docstring for TargetedHNetwork"""
	def __init__(self, config):
		super(TargetedHNetwork, self).__init__(config)

		self.satt_layer = AttNet(config, config.hidden_dim)
		self.datt_layer = AttNet(config, config.hidden_dim * 2)

		self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

	def embed_tl(self, target, lead):

		target_embed = self.embed_target(target)
		# target_embed =  self.wembed(target).squeeze(1)
		#leads(batch_size, max_num_doc, max_seqlen)
		#nleads(batch_size, max_num_doc)
		#lead_nwords(batch_size, max_num_doc, max_seqlen)

		leads, nleads, lead_nwords = lead
		# (batch_size, max_num_doc)
		# lead_mask = sequence_mask(nleads)
		# (batch_size, max_num_doc, max_seqlen)
		lead_word_mask = sequence_mask(lead_nwords, device = self.config.gpu)
		leads = leads.view(-1, leads.size(-1))
		#(batch_size * max_num_doc, max_seqlen, embed_dim)
		leads_embeded = self.wembed(leads)

		#(batch_size * max_num_doc, hidden_dim)
		_, lead_reprs = self.w2s(leads_embeded, init = target_embed,\
		 mask = lead_word_mask.view(-1, lead_word_mask.size(-1), 1))
		
		#dcmts(batch_size, max_num_doc, max_num_sent, max_seqlen)
		#ndocs(batch_size)
		#nsents(batch_size, max_num_doc)
		#dcmt_nwords(batch_size, max_num_doc, max_num_sent)
		lead_reprs = lead_reprs.view(-1, lead_nwords.size(1), lead_reprs.size(-1))

		return target_embed, lead_reprs

	def embed_body(self, dcmt, lead_reprs, init = None):

		dcmts, ndocs, nsents, dcmt_nwords = dcmt

		doc_mask = sequence_mask(ndocs, device = self.config.gpu)
		doc_sent_mask = sequence_mask(nsents, device = self.config.gpu)
		doc_word_mask = sequence_mask(dcmt_nwords, device = self.config.gpu)

		batch_size, max_num_doc, max_num_sent, max_seqlen = dcmts.size()
		dcmts_reshaped = dcmts.view(-1, max_seqlen)

		#(batch_size * max_num_doc * max_num_sent, max_seqlen, embed_dim)
		dcmts_embeded = self.wembed(dcmts_reshaped)

		#(batch_size * max_num_doc * max_num_sent, hidden_dim * 2)
		_, sent_reprs = self.w2s(dcmts_embeded, init = init, mask = \
			doc_word_mask.view(-1, doc_word_mask.size(-1), 1))


		#(batch_size * max_num_doc, max_num_sent, hidden_dim * 2)
		sent_reprs = sent_reprs.view(-1, max_num_sent, sent_reprs.size(-1))
		doc_sent_mask = doc_sent_mask.view(-1, doc_sent_mask.size(-1), 1)
		
		#(batch_size * max_num_doc, max_num_sent, hidden_dim )
		ctx_sent_reprs, _ = self.s2d(sent_reprs, mask = doc_sent_mask)#, init = init)
		lead_reprs_reshaped = lead_reprs.view(-1, lead_reprs.size(-1))
		doc_reprs = self.satt_layer(\
			 ctx_sent_reprs, lead_reprs_reshaped, doc_sent_mask)

		doc_reprs = doc_reprs.view(-1, max_num_doc, doc_reprs.size(-1))

		return doc_reprs, doc_mask


	def embed_doc(self, target, lead, dcmt):	

		tgt_embeded, lead_reprs = self.embed_tl(target, lead)
		doc_reprs, doc_mask = self.embed_body(dcmt, lead_reprs, init = tgt_embeded)

		doc_reprs = torch.cat([lead_reprs, doc_reprs], dim = -1)
		doc_reprs = self.datt_layer(facts = doc_reprs, fact_mask = doc_mask.unsqueeze(-1))
		

		return doc_reprs