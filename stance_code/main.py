import os
import sys
sys.path.append('..')

import random
from torchtext.vocab import GloVe#, Vectors, CharNGram
from torchtext.data import TabularDataset, \
LabelField, Field, NestedField, Iterator,Pipeline, BucketIterator

import numpy as np
import logging
from argparse import ArgumentParser

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, \
create_supervised_evaluator

from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
from new_model import *
from baselines import *

def parse_args():
	parser = ArgumentParser(description='pytorch/stance')
	parser.add_argument('--model', type = str, default = "hn")
	parser.add_argument('--train_file', type = str, default = 'train_2068.tsv')
	parser.add_argument('--test_file', type = str, default = 'test_524.tsv')
	parser.add_argument('--kernel_sizes', type = str, default = '1,2,3,4')
	parser.add_argument('--num_filters', type = int, default = 50)
	parser.add_argument('--embed_dim', type = int, default = 50	)
	parser.add_argument('--hidden_dim', type = int, default = 50	)
	parser.add_argument('--num_classes', type = int, default = 3)
	parser.add_argument('--num_layers', type = int, default = 1)
	parser.add_argument('--log_interval',type = int, default = 10)
	parser.add_argument('--batch_size', type = int, default = 64)
	parser.add_argument('--warm_up', type = int, default = 200)
	parser.add_argument('--epochs', type = int, default = 100)
	parser.add_argument('--gpu', type = int, default = -1)
	parser.add_argument('--seed', type = int, default = 1234)
	parser.add_argument('--lr', type = float, default = 3e-4)
	parser.add_argument('--dropout', type = float, default = 0.1)
	parser.add_argument('--save', action = "store_true")
	parser.add_argument('--resume', type = str, default = 'none')
	args = parser.parse_args()

	return args

def init_logger(config):
  fmt = "%(asctime)s - %(levelname)s - %(message)s"
  handler = logging.StreamHandler()
  # fname = get_model_name(config) + ".log"
  # handler = logging.FileHandler(fname)
  handler.setFormatter(logging.Formatter(fmt))

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logger.addHandler(handler)

  return logger


def get_model_name(config):
	model_name = [config.model,\
	 'b' + str(config.batch_size), 'lr' + str(config.lr),\
	 'dp' + str(config.dropout), str(os.getpid())]

	return '_'.join(model_name)

def get_model(config):
	model_name = config.model
	if model_name in ("HN", "HAN"):
		return HNetwork(config)
	elif model_name == "THAN": #TEND-C
		return TargetedHNetwork(config)
	elif model_name == "HL": # word-word alignment
		return HLNetwork(config)
	elif model_name == "TE": #TEND-T
		return TransEncoder(config)
	elif model_name == "CE": # conditional encoding
		return ConditionEncode(config)			
	else:
		raise Exception("%s not implemented!" % config.model)

def load_dataset(config, device):


	label_dict = {"observing":0, "against":1, "for":2 }
	LABEL = Field(use_vocab = False, sequential = False,\
	 dtype = torch.long, preprocessing = lambda x: label_dict[x.strip()])
	
	SEQ = Field(dtype = torch.long, lower = True, batch_first = True,\
	 preprocessing = lambda x:x[:45], include_lengths = True)
	SENT = Field(dtype = torch.long, lower = True, batch_first = True,\
	 preprocessing = lambda x:x[:45], include_lengths = False)

	DOC = NestedField(SENT, tokenize = lambda s:s.strip().split(' </s> '), \
	 preprocessing = lambda s:[x for x in s[:45] if x], dtype = torch.long,\
	 include_lengths = True)

	fields = [('label', LABEL), ('claim', SEQ), ('hline', SEQ),\
	 ('abst', SEQ), ('body', DOC)]
	
	train, test = TabularDataset.splits(path="../stance_data/", format = "tsv",\
	 fields = fields, train = config.train_file, test = config.test_file)
	train, val = train.split(split_ratio = 0.80)

	vectors = GloVe(name = "6B", dim = config.embed_dim, cache = '/users4/jwduan/vectors/')		
	DOC.build_vocab(train, val, test, vectors = vectors)


	SEQ.build_vocab()
	SEQ.vocab = DOC.vocab

	config.vocab_size = len(DOC.vocab)
	train_loader, val_loader, test_loader = Iterator.splits((train, val, test),\
	 batch_sizes = (config.batch_size, 256, 256), sort_key = lambda x:len(x.body), sort = True,
	  device = device, shuffle = True, repeat = False)

	return (train_loader, val_loader, test_loader), DOC.vocab.vectors

def evaluate(engine, loader):
	engine.run(loader)
	metrics = engine.state.metrics
	avg_accuracy = metrics['accuracy']
	avg_nll = metrics.get('nll', 0.)

	return avg_nll, avg_accuracy

def run_test(model, evaler, loader, resume_path, logger):

	# resume_path =  resume
	logger.info("====> loading model from %s" % resume_path)
	model.load_state_dict(torch.load(resume_path))
	_, test_acc = evaluate(evaler, loader)
	logger.warn("test set acc {:.4f}".format(test_acc))


def run(args):

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	# torch.backends.cudnn.deterministic = True
	device = torch.device("cuda:%s" % args.gpu\
	 if torch.cuda.is_available() and args.gpu != -1 else "cpu")

	logger = init_logger(args)
	logger.info(' '.join(sys.argv))

	(train_loader, val_loader, \
		test_loader), vectors = load_dataset(args, device)
	model = get_model(args)

	testor = create_supervised_evaluator(model,
							metrics = {'accuracy': Accuracy()}, device = device)

	if args.resume != "none":
		run_test(model, testor, test_loader, args.resume, logger)
		return
	
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	model.embed = nn.Embedding.from_pretrained(vectors, freeze = False)

	trainer = create_supervised_trainer(model, optimizer,\
	 F.nll_loss, device = device)

	evaluator = create_supervised_evaluator(model,
							metrics = {
							'accuracy': Accuracy(),\
							'nll': Loss(F.nll_loss) }, device = device)

	def score_function(engine):
		val_loss = engine.state.metrics['nll']
		return -val_loss

	def acc_score_function(engine):
		return engine.state.metrics['accuracy']

	handler = EarlyStopping(patience = 50, \
		score_function = score_function, trainer = trainer)
	evaluator.add_event_handler(Events.COMPLETED, handler)

	if args.save:
		chkpt_handler = ModelCheckpoint('./models', get_model_name(args),\
		 score_function = acc_score_function, n_saved = 1, create_dir = True,\
		 atomic = True, save_as_state_dict = True, score_name = "acc")
		evaluator.add_event_handler(Events.COMPLETED, chkpt_handler, {'model':model})

	setattr(trainer, "best_acc", 0.)

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_loss(engine):
		num_iter = engine.state.iteration
		logger.info("Epoch {} Iteration {} Loss: {:.4f}"
				"".format(engine.state.epoch, num_iter, engine.state.output))

	@trainer.on(Events.ITERATION_COMPLETED)
	def log_validation_results(engine):

		num_iter = engine.state.iteration

		if num_iter % args.log_interval == 0:
		# if num_iter >= args.warm_up and num_iter % args.log_interval == 0:
			avg_nll, avg_accuracy = evaluate(evaluator, val_loader)

			logger.info("Iteration {}, Avg accuracy: {:.4f} Avg loss: {:.4f}"
				.format(num_iter, avg_accuracy, avg_nll))

			if avg_accuracy > engine.best_acc:
				logger.warn("best validation accuracy {:.4f}".format(avg_accuracy))
				engine.best_acc = avg_accuracy

	trainer.run(train_loader, max_epochs = args.epochs)

if __name__ == '__main__':

	args = parse_args()
	run(args)
