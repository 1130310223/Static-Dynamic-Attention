import sys
import os
import random
import re
import time
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from static_model import StaticModel
from dyna_model import DynamicModel
from data_utils import *

def init_command_line(argv):
	from argparse import ArgumentParser
	usage = "train"
	description = ArgumentParser(usage)
	description.add_argument("--train_data", type=int, default=1)

	description.add_argument("--max_context_size", type=int, default=9)
	description.add_argument("--batch_size", type=int, default=80)
	description.add_argument("--hidden_size", type=int, default=512)
	description.add_argument("--max_senten_len", type=int, default=15)
	description.add_argument("--type_model", type=int, default=1)

	description.add_argument("--lr", type=float, default=0.001)
	description.add_argument("--weight_decay", type=float, default=1e-5)
	description.add_argument("--dropout", type=float, default=0.5)
	description.add_argument("--epochs", type=int, default=10)
	description.add_argument("--teach_forcing", type=int, default=1)

	description.add_argument("--shuffle", type=int, default=1)
	description.add_argument("--print_every", type=int, default=200)
	description.add_argument("--save_model", type=int, default=1)
	description.add_argument("--weights", type=str, default=None)
	return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print ("Configure:")
if opts.train_data:
	print (" Opensubtitle dataset for training")
	train_file = "./data/open_train_sessions.txt"
	w2v_file = "./data/open_train_all_100e.w2v"
else:
	print (" Ubuntu dataset for training")
	train_file = "./data/ubuntu_train_sessions.txt"
	w2v_file = "./data/ubuntu_train_all_100e.w2v"
print (" train_file:",train_file)
print (" w2v_file:",w2v_file)

print (" max_context_size:",opts.max_context_size)
print (" batch_size:",opts.batch_size)
print (" hidden_size:",opts.hidden_size)
print (" max_senten_len:",opts.max_senten_len)
if opts.type_model:
	print (" static model")
else:
	print (" dynamic model")
print (" learning rate:",opts.lr)
print (" weight_decay:",opts.weight_decay)
print (" dropout:",opts.dropout)
print (" epochs:",opts.epochs)
print (" teach_forcing:",opts.teach_forcing)

print (" shuffle:",opts.shuffle)
print (" print_every:",opts.print_every)
print (" save_model:",opts.save_model)
print (" weights:",opts.weights)
print ("")

def train_batch(reply_tensor_batch,contexts_tensor_batch,pad_matrix_batch,model,model_optimizer,criterion,ini_idx):
	loss = 0
	model_optimizer.zero_grad()

	list_pred = model(reply_tensor_batch,contexts_tensor_batch,pad_matrix_batch,ini_idx)
	for idx,reply_tensor in enumerate(reply_tensor_batch):
		loss_s = criterion(list_pred[idx],Variable(reply_tensor).cuda())
		loss += loss_s

	loss.backward()
	model_optimizer.step()
	
	return loss.data[0]

def train_model(word2index,ini_idx,corpus_pairs,model,model_optimizer,criterion,epochs,
				batch_size,max_senten_len,max_context_size,print_every,save_model,shuffle):
	print ("start training...")
	model.train()
	state_loss = 10000.0
	for ei in range(epochs):
		print ("Iteration {}: ".format(ei+1))
		epoch_loss = 0
		every_loss = 0
		t0 = time.time()
		pairs_batches,num_batches = buildingPairsBatch(corpus_pairs,batch_size,shuffle=shuffle)
		print ("num_batches:",num_batches)

		idx_batch = 0
		for reply_tensor_batch, contexts_tensor_batch, pad_matrix_batch in getTensorsPairsBatch(word2index,pairs_batches,max_context_size):
			loss = train_batch(reply_tensor_batch,contexts_tensor_batch,pad_matrix_batch,model,model_optimizer,criterion,ini_idx)
			epoch_loss += loss
			every_loss += loss
			if (idx_batch+1)%print_every == 0:
				every_avg_loss = every_loss/(max_senten_len*(idx_batch+1))
				print ("{} batches finished, avg_loss:{}".format(idx_batch+1, every_avg_loss))
			idx_batch += 1

		epoch_avg_loss = epoch_loss/(max_senten_len*num_batches)
		print ("epoch_avg_loss:",epoch_avg_loss)
		if save_model and epoch_avg_loss < state_loss:
			print ("save model...")
			if opts.type_model:
				if opts.train_data:
					torch.save(model.state_dict(), "./static_open_parameters_IterEnd")
				else:
					torch.save(model.state_dict(), "./static_ubu_parameters_IterEnd")
			else:
				if opts.train_data:
					torch.save(model.state_dict(), "./dyna_open_parameters_IterEnd")
				else:
					torch.save(model.state_dict(), "./dyna_ubu_parameters_IterEnd")
			state_loss = epoch_avg_loss

		print ("Iteration time:",time.time()-t0)
		print ("=============================================" )
		print ("")

if __name__ == '__main__':
	ini_char = '</i>'
	unk_char = '<unk>'
	t0 = time.time()
	print ("loading word2vec...")
	ctable = W2vCharacterTable(w2v_file,ini_char,unk_char)
	print(" dict size:",ctable.getDictSize())
	print (" emb size:",ctable.getEmbSize())
	print ("")

	ctable,corpus_pairs = readingData(ctable,train_file,opts.max_senten_len,opts.max_context_size)
	print (time.time()-t0)
	print ("")

	if opts.type_model:
		model = StaticModel(ctable.getDictSize(),ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.dropout,
						opts.max_senten_len,opts.teach_forcing).cuda()
	else:
		model = DynamicModel(ctable.getDictSize(),ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.dropout,
						opts.max_senten_len,opts.teach_forcing).cuda()

	if opts.weights != None:
		print ("load weights...")
		model.load_state_dict(torch.load(opts.weights))
	else:
		model.init_parameters(ctable.getEmbMatrix())

	model_optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
	criterion = nn.NLLLoss()
	print ("")

	word2index = ctable.getWord2Index()
	ini_idx = word2index[ini_char]
	train_model(word2index,ini_idx,corpus_pairs,model,model_optimizer,criterion,opts.epochs,opts.batch_size,
				opts.max_senten_len,opts.max_context_size,opts.print_every,opts.save_model,opts.shuffle)
	print ("")
