 #-*- coding: utf-8 -*-
import sys
import os
import random
import re
import time
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
#sys.path.append('../')
from hybrid_model import Seq2Seq
from hybrid_data_utils import *

if not os.path.exists('./result/'):
	os.mkdir('./result/')

sub = '-'*20
def init_command_line(argv):
	from argparse import ArgumentParser
	usage = "seq2seq"
	description = ArgumentParser(usage)
	description.add_argument("--w2v_path", type=str, default="/users3/yfwang/rg_data/w2v/opensubtitle/")
	description.add_argument("--w2v", type=str, default="train_all_200e.w2v")
	
	description.add_argument("--max_context_size", type=int, default=9)
	description.add_argument("--batch_size", type=int, default=80)
	description.add_argument("--hidden_size", type=int, default=512)
	description.add_argument("--max_senten_len", type=int, default=15)

	description.add_argument("--dropout", type=float, default=0.5)

	description.add_argument("--teach_forcing", type=int, default=1)
	description.add_argument("--print_every", type=int, default=100, help="print every batches when training")
	description.add_argument("--weights", type=str, default=None)
	return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print ("Configure:")
print (" w2v:",os.path.join(opts.w2v_path,opts.w2v))

print (" max_context_size:",opts.max_context_size)
print (" batch_size:",opts.batch_size)
print (" hidden_size:",opts.hidden_size)
print (" max_senten_len:",opts.max_senten_len)

print (" dropout:",opts.dropout)

print (" teach_forcing:",opts.teach_forcing)
print (" print_every:",opts.print_every)
print (" weights:",opts.weights)
print ("")

def readingTestCorpus(test_file_path):
	print ("reading...")
	test_file = open(test_file_path,'r')
	list_pairs = []
	tmp_pair = []
	for line in test_file:
		line = line.strip('\n')
		if line == sub:
			list_pairs.append(tmp_pair)
			tmp_pair = []
		else:
			tmp_pair.append(line)
	test_file.close()

	test_contexts = []
	test_replys = []
	max_con_size =  0
	min_con_size = 10000
	for pair in list_pairs:
		if len(pair) >= 3:
			test_contexts.append(pair[0:-1])
			test_replys.append(pair[-1])
			max_con_size = max(len(pair[0:-1]),max_con_size)
			min_con_size = min(len(pair[0:-1]),min_con_size)
		else:
			pass
	print (max_con_size)
	print (min_con_size)
	return test_contexts,test_replys

def preProcess(word2index,test_contexts,unk_char,ini_char,max_senten_len,max_context_size):
	print ("preprocessing...")
	filter_test_contexts = []
	for context in test_contexts:
		filter_context = [filteringSenten(word2index,senten,unk_char,ini_char) for senten in context]
		filter_test_contexts.append(filter_context)

	padded_test_pairs = []
	for context in filter_test_contexts:
		pad_list = [0]*len(context)
		if len(context) <= max_context_size:
			pad_list = [1]*(max_context_size-len(context)) + pad_list
			context = ['<unk>']*(max_context_size-len(context)) + context
		else:
			pad_list = pad_list[-max_context_size:]
			context = context[-max_context_size:]
		padded_context = [paddingSenten(senten,max_senten_len) for senten in context]
		padded_test_pairs.append([padded_context,pad_list])

	return padded_test_pairs


# 读入一个句子的list，构建batch后进行预测
def predictSentences(index2word,unk_char,ini_char,ini_idx,model,test_pairs,
					print_every,batch_size,max_senten_len,max_context_size):
	model.eval()
	#构造batch的list
	pairs_batches,num_batches = buildingPairsBatch(test_pairs,batch_size,shuffle=False)
	print ("")
	print ("num of batch:",num_batches)
	
	predict_sentences = []
	idx_batch = 0
	for contexts_tensor_batch, pad_matrix_batch in getTensorsContextPairsBatch(word2index,pairs_batches,max_context_size):
		predict_batch = model.predict(contexts_tensor_batch,index2word,pad_matrix_batch,ini_idx,sep_char='\t')
		predict_sentences.extend(predict_batch)
		if (idx_batch+1)%print_every == 0:
			print ("{} batches finished".format(idx_batch+1))
		idx_batch += 1

	predict_sentences = predict_sentences[0:len(test_pairs)]
	return predict_sentences

if __name__ == '__main__':
	ini_char = '</i>'
	unk_char = '<unk>'
	t0 = time.time()
	print ("loading word2vec...")
	ctable = W2vCharacterTable(os.path.join(opts.w2v_path,opts.w2v),ini_char,unk_char)
	print(" dict size:",ctable.getDictSize())
	print (" emb size:",ctable.getEmbSize())
	print (time.time()-t0)
	print ("")

	seq2seq = Seq2Seq(ctable.getDictSize(),ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.dropout,
					opts.max_senten_len,opts.teach_forcing).cuda()

	if opts.weights != None:
		print ("load model parameters...")
		seq2seq.load_state_dict(torch.load(opts.weights))
	else:
		print ("No model parameters!")
		exit()

	test_file_path = "/users3/yfwang/rg_data/corpus/opensubtitle/test_sessions.txt"
	test_contexts,test_replys = readingTestCorpus(test_file_path)
	print ("len(test_contexts):",len(test_contexts))
	print ("len(test_replys):",len(test_replys))

	word2index = ctable.getWord2Index()
	test_pairs = preProcess(word2index,test_contexts,unk_char,ini_char,opts.max_senten_len,opts.max_context_size)
	print ("len(test_pairs):",len(test_pairs))
	'''test_pair = test_pairs[100]
	test_context = test_pair[0]
	pad_list = test_pair[1]

	for senten in test_context:
		print senten
	print pad_list'''
	
	print ("start predicting...")
	ini_idx = word2index[ini_char]
	predict_sentences = predictSentences(ctable.getIndex2Word(),unk_char,ini_char,ini_idx,seq2seq,test_pairs,
									opts.print_every,opts.batch_size,opts.max_senten_len,opts.max_context_size)

	print ("writing...")
	pred_res_file = open("./result/open_pred_res_hybrid_t1",'w')
	pred_ans_file = open("./result/open_pred_ans_hybrid_t1",'w')
	for idx,senten in enumerate(predict_sentences):
		test_context = test_contexts[idx]
		for test_post in test_context:
			pred_res_file.write(test_post+'\n')
		pred_res_file.write(senten+'\n')
		pred_res_file.write(sub+'\n')
		senten_l = [c for c in senten.split('\t') if c != '</s>']
		pred_ans_file.write(' '.join(senten_l)+' __eou__'+'\n')

	pred_res_file.close()
	pred_ans_file.close()
	print ("end")
	
