import os
import sys
sys.path.append('..')
import codecs
import random

# from torchtext.vocab import Vectors,GloVe
from torchtext.data import TabularDataset, Field, NestedField, BucketIterator
from NestNestedField import NestNestedField

import numpy as np
import logging
from argparse import ArgumentParser

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from microf1 import Microf1
from auc import AUC
from ignite.handlers import EarlyStopping, ModelCheckpoint
from utils import MyVectors

from baselines import HNetwork
from model import TargetedHNetwork
from new_model import TransEncoder


def parse_args():
	parser = ArgumentParser(description='targeted-document')
	parser.add_argument('--model', type = str, default = "HN")
	parser.add_argument('--mode', type = str, default = "avg")
	parser.add_argument('--train_file', type = str, default = 'train.txt')
	parser.add_argument('--dev_file', type = str, default = 'dev.txt')
	parser.add_argument('--test_file', type = str, default = 'test.txt')
	parser.add_argument('--embed_dim', type = int, default = 100)
	parser.add_argument('--hidden_dim', type = int, default = 100)
	parser.add_argument('--num_layers', type = int, default = 1)
	parser.add_argument('--num_classes', type = int, default = 2)
	parser.add_argument('--log_interval',type = int, default = 10)
	parser.add_argument('--batch_size', type = int, default = 32)
	parser.add_argument('--epochs', type = int, default = 10)
	parser.add_argument('--gpu', type = int, default = -1)
	parser.add_argument('--lr', type = float, default = 0.0005)
	parser.add_argument('--dropout', type = float, default = 0.1)
	parser.add_argument('--save', action = "store_true")
	parser.add_argument('--resume', type = str, default = 'none')
	args = parser.parse_args()

	return args

def init_logger():
	fmt = "%(asctime)s - %(levelname)s - %(message)s"
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter(fmt))

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)

	return logger


def get_model(config):
	model_name = config.model

	if model_name == "THN":
		return TargetedHNetwork(config)
	elif model_name == "TE":
		return TransEncoder(config)
	else:
		raise Exception("%s not implemented!" % model_name)

def get_model_name(config):

	model_name = [config.model, config.mode, config.batch_size, config.embed_dim,\
		config.hidden_dim, config.lr, config.dropout, "epoch",\
		config.epochs, os.getpid()]
	return "_".join(map(str, model_name))


def load_dataset(config, device):

	LABEL = Field(sequential = False, dtype = torch.long, use_vocab = False,\
	 batch_first = True, preprocessing = lambda x:1 if float(x) > 0. else 0)
	TARGET = Field(batch_first = True, lower = True, dtype = torch.long,
		preprocessing = lambda x:x[0].split('_'), include_lengths = True)

	TEXT = Field(dtype = torch.long, lower = True, batch_first = True,\
	 preprocessing = lambda x:x[:50])# [w for w in x if w not in stopwords_set][:50])

	LEADS = NestedField(TEXT, dtype = torch.long, include_lengths = True,\
	 tokenize = lambda s: s.split('</s>'), preprocessing = lambda x:x[-5:])

	DOC = NestedField(TEXT, dtype = torch.long, include_lengths = True,\
	 tokenize = lambda s: s.split('</s>'), preprocessing = lambda x:[s for s in x[1:50] if s])
	DOCS = NestNestedField(DOC, dtype = torch.long, include_lengths = True,\
	 tokenize = lambda s: s.split('</p>'), preprocessing = lambda x:x[-5:])

	fields = [('label', LABEL), ('target',TARGET), ('leads', LEADS), ('docs', DOCS)]
	train, val, test = TabularDataset.splits(path="../abrt_data/", format = "tsv", \
		fields = fields, train = config.train_file, validation = config.dev_file, test = config.test_file)


	TARGET.build_vocab(train, val, test)
	DOCS.build_vocab(train, val, test)


	config.wvocab_size = len(DOCS.vocab)
	config.tvocab_size = len(TARGET.vocab)
 	# sort = False,
	train_loader, val_loader, test_loader = BucketIterator.splits((train, val, test),\
	 sort_key = lambda x: len(x.docs), sort = True, batch_sizes = (config.batch_size, 32, 32),\
		device = device, repeat = False)
	return (train_loader, val_loader, test_loader)

def evaluate(engine, loader):
	engine.run(loader)
	metrics = engine.state.metrics
	
	avg_nll = metrics['nll']
	microf1 = metrics.get('microf1', 0.)
	auc = metrics.get('auc', (0., 0.))

	return avg_nll, microf1, auc

def run_test(model, evaler, loader, resume_path, logger):
	logger.info("====> loading model from %s" % resume_path)
	model.load_state_dict(torch.load(resume_path))
	_, microf1, test_auc = evaluate(evaler, loader)
	logger.warning("test set microf1 {:.4f},\
	 auc {:.4f} {:.4f}".format(microf1, test_auc[0], test_auc[1]))

def run(args):

	random.seed(1234)
	torch.manual_seed(1234)
	# torch.backends.cudnn.deterministic = True

	device = torch.device("cuda:%s" % args.gpu\
	 if torch.cuda.is_available() and args.gpu != -1 else "cpu")

	logger = init_logger()
	logger.info(' '.join(sys.argv))


	train_loader, val_loader, test_loader = load_dataset(args, device)
	
	model = get_model(args)

	testor = create_supervised_evaluator(model,
							metrics = {
							'microf1': Microf1(),\
							'auc': AUC(),\
							'nll': Loss(F.nll_loss) }, device = device)
	
	if args.resume != "none":
		run_test(model, testor, test_loader, args.resume, logger)
		return	

	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	trainer = create_supervised_trainer(\
		model, optimizer, F.nll_loss, device = device)

	evaluator = create_supervised_evaluator(model,
							metrics = {
							'microf1': Microf1(),\
							'auc': AUC(),\
							'nll': Loss(F.nll_loss) }, device = device)

	def score_function(engine):
		val_loss = engine.state.metrics['nll']
		return -val_loss

	def microf1_score_function(engine):
		return engine.state.metrics["microf1"]

	handler = EarlyStopping(patience = 50, \
		score_function = score_function, trainer = trainer)
	evaluator.add_event_handler(Events.COMPLETED, handler)

	if args.save:
		chkpt_handler = ModelCheckpoint('./model', get_model_name(args),\
			score_function = microf1_score_function, n_saved = 1, create_dir = True,\
			atomic = True, save_as_state_dict = True, score_name = "microf1")
		evaluator.add_event_handler(Events.COMPLETED, chkpt_handler, {'model':model})

	setattr(trainer, "best_microf1", 0.)
	model_name = get_model_name(args)

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_loss(engine):
		num_iter = engine.state.iteration
		logger.info("Epoch {} Iteration {} Loss: {:.4f}"
				"".format(engine.state.epoch, num_iter, engine.state.output))

	@trainer.on(Events.ITERATION_COMPLETED)
	def log_validation_results(engine):

		num_iter = engine.state.iteration

		if num_iter >= 1000 and num_iter % args.log_interval == 0:
			avg_nll, microf1, auc = evaluate(evaluator, val_loader)
			
			logger.info("Iteration {}, microf1: {:.4f} Avg loss: {:.4f}"
				.format(num_iter, microf1, avg_nll))

			if microf1 > engine.best_microf1:
				logger.warning("best validation microf1 {:.4f}, auc ({:.4f},{:.4f})".format(\
					microf1, auc[0], auc[1]))
				engine.best_microf1 = microf1

	trainer.run(train_loader, max_epochs = args.epochs)


if __name__ == '__main__':

	args = parse_args()
	run(args)
