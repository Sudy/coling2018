import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import *

class AttNet(nn.Module):
	"""docstring for AttNet"""
	def __init__(self, config, input_dim):
		super(AttNet, self).__init__()
		self.config = config
		self.affine = nn.Linear(input_dim, 1)

	#query(batch_size, hidden_dim)
	#facts(batch_size, max_seqlen, hidden_dim)
	#fact_mask(batch_size, max_seqlen, 1)
	#return (batch_size, hidden_dim)

	def forward(self, query, facts, fact_mask):

		if query is not None:
			max_seqlen = facts.size(1)
			query = query.unsqueeze(1).repeat(1, max_seqlen, 1)
			# print(query.size(), facts.size())
			#(batch_size, max_seqlen, hidden_dim * 2)
			output = torch.cat([query, facts], dim = -1)
		else:
			output = facts

		#(batch_size, max_seqlen, 1)
		weights = self.affine(output)
		norm_weights = softmax_with_mask(weights, mask = fact_mask, dim = 1)

		return torch.sum(norm_weights * facts, dim = 1), norm_weights

#sequential representation
class SequentialRepr(nn.Module):
	def __init__(self, config, input_dim = 100, mode = 'lstm_avg'):
		super(SequentialRepr, self).__init__()
		self.encoder = nn.LSTM(input_size = input_dim, batch_first = True, \
			hidden_size = config.hidden_dim,	bidirectional = True, \
			num_layers = config.num_layers, dropout = config.dropout)

		self.mode = mode
		self.config = config
	
	#text_embed(batch_size, seqlen, embed_dim)
	#init (batch_size, embed_dim)
	#mask(batch_size, seqlen, 1)

	def forward(self, text_embed, init = None, mask = None):

		# self.encoder.flatten_parameters()
		# (batch_size, seq_len, num_direction * hidden_size)
		if init is None:
			hiddens, (h_n, c_n) = self.encoder(text_embed)
		else:
			batch_size = text_embed.size(0) // init.size(0)
			h0 = init.unsqueeze(0).repeat(2, batch_size, 1)
			c0 = Variable(torch.zeros(h0.size()).float()).cuda()
			hiddens, (h_n, c_n) = self.encoder(text_embed, (h0, c0))

		hiddens = torch.chunk(hiddens, chunks = 2, dim = -1)
		hiddens = torch.mean(torch.cat(\
			[h.unsqueeze(-1) for h in hiddens], dim = -1), dim = -1)

		# print(hiddens.size(), mask.size())
		if self.mode == 'lstm_ctx':
			return hiddens, h_n
		
		elif self.mode == 'lstm_avg':
			if mask is not None:
				output = torch.sum(hiddens * mask, dim = 1) / (torch.sum(mask, dim = 1) + 10-8)
			else:
				output = torch.mean(hiddens, dim = 1, keepdim = False)

		return hiddens, output


class BaseHNetwork(nn.Module):
	"""docstring for TargetedHNetwork"""
	def __init__(self, config):
		super(BaseHNetwork, self).__init__()

		self.wembed = nn.Embedding(config.wvocab_size, config.embed_dim)
		self.tembed = nn.Embedding(config.tvocab_size, config.embed_dim)

		self.w2s = SequentialRepr(config, config.embed_dim, "lstm_avg")
		self.s2d = SequentialRepr(config, config.hidden_dim, 'lstm_ctx')
		self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

	def embed_doc(self, batch):
		raise Exception('Not implemented!')

	def forward(self, batch):
		dcmt_reprs = self.embed_doc(batch)
		pred = self.fc(dcmt_reprs)
		return F.log_softmax(pred, dim = -1)

class TargetedHNetwork(BaseHNetwork):
	"""docstring for TargetedHNetwork"""
	def __init__(self, config):
		super(TargetedHNetwork, self).__init__(config)
		self.satt_layer = AttNet(config, config.hidden_dim * 2)
		self.datt_layer = AttNet(config, config.hidden_dim * 2)

	def embed_doc(self, batch):		
		#target(batch_size, seqlen)
		#target_nwords(batch_size)
		target, target_nwords= batch.target
		#(batch_size, seqlen)
		target_word_mask = sequence_mask(target_nwords)
		#(batch_size, seqlen, embed_dim)
		target_embed = self.tembed(target)
		target_word_mask = target_word_mask.unsqueeze(-1)
		target_embed = torch.sum(target_embed * target_word_mask, \
			dim = 1) / torch.sum(target_word_mask, dim = 1)

		#leads(batch_size, max_num_doc, max_seqlen)
		#nleads(batch_size, max_num_doc)
		#lead_nwords(batch_size, max_num_doc, max_seqlen)
		leads, nleads, lead_nwords = batch.leads
		# (batch_size, max_num_doc)
		# lead_mask = sequence_mask(nleads)
		# (batch_size, max_num_doc, max_seqlen)
		lead_word_mask = sequence_mask(lead_nwords)

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
		dcmts, ndocs, nsents, dcmt_nwords = batch.docs

		doc_mask = sequence_mask(ndocs)
		doc_sent_mask = sequence_mask(nsents)
		doc_word_mask = sequence_mask(dcmt_nwords)

		batch_size, max_num_doc, max_num_sent, max_seqlen = dcmts.size()
		dcmts_reshaped = dcmts.view(-1, max_seqlen)
		#(batch_size * max_num_doc * max_num_sent, max_seqlen, embed_dim)
		dcmts_embeded = self.wembed(dcmts_reshaped)

		#(batch_size * max_num_doc * max_num_sent, hidden_dim * 2)
		_, sent_reprs = self.w2s(dcmts_embeded, init = target_embed, mask = \
			doc_word_mask.view(-1, doc_word_mask.size(-1), 1))

		#(batch_size * max_num_doc, max_num_sent, hidden_dim * 2)
		sent_reprs = sent_reprs.view(-1, max_num_sent, sent_reprs.size(-1))
		doc_sent_mask = doc_sent_mask.view(-1, doc_sent_mask.size(-1), 1)
		
		# print(sent_reprs.size())
		ctx_sent_reprs, _ = self.s2d(sent_reprs, init = target_embed)
		# ctx_sent_reprs, _ = self.s2d(sent_reprs)

		doc_reprs = self.satt_layer(lead_reprs, ctx_sent_reprs, doc_sent_mask)
		lead_reprs = lead_reprs.view(-1, max_num_doc, lead_reprs.size(-1))
		doc_reprs = doc_reprs.view(-1, max_num_doc, doc_reprs.size(-1))
		doc_reprs = torch.cat([lead_reprs, doc_reprs], dim = -1)
		doc_mask = doc_mask.unsqueeze(-1)
		doc_reprs = self.datt_layer(None, doc_reprs, doc_mask)
		return doc_reprs


