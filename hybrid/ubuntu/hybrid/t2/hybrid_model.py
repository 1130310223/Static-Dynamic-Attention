# -*- coding: utf-8 -*-
import sys
import math
import time
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

class Encoder(nn.Module):
	def __init__(self, emb_size, hidden_size):
		super(Encoder,self).__init__()
		self.emb_size = emb_size
		self.hidden_size = hidden_size

		self.gru = nn.GRU(self.emb_size,self.hidden_size,batch_first=True)

	def forward(self, input, hidden):
		'''input&output: (batch_size,max_senten_len,emb_size)
		hidden: (1,batch_size,hidden_size)'''
		output,hidden = self.gru(input,hidden)

		return hidden

class TopicAttention(nn.Module):
	def __init__(self, hidden_size):
		super(TopicAttention, self).__init__()
		self.hidden_size = hidden_size

		self.topic_W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.topic_U = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.topic_v = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, inputs, hidden, pad_matrix):
		'''hidden&inputs[i]: (1,batch_size,hidden_size)
		len(input): context_size, pad_matrix: (batch_size,context_size)
		topic_attn_weights: (batch_size,context_size)'''
		topic_attn_hidden = torch.mm(hidden.squeeze(0),self.topic_W)
		list_weight = []
		for input in inputs:
			# weights: (batch_size,1)
			weight = torch.mm(self.tanh(topic_attn_hidden+torch.mm(input.squeeze(0),self.topic_U)),self.topic_v)
			list_weight.append(weight)
		# context中padding的句子不参与attention的计算
		topic_attn_weights = torch.cat(tuple(list_weight),1).masked_fill_(Variable(pad_matrix.cuda()),-100000)
		topic_attn_weights = self.softmax(topic_attn_weights)

		return topic_attn_weights

class Decoder(nn.Module):
	def __init__(self, decoder_dict_size, hidden_size, emb_size):
		super(Decoder, self).__init__()
		self.decoder_dict_size = decoder_dict_size
		self.hidden_size = hidden_size
		self.emb_size = emb_size

		self.dyna_W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.dyna_U = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
		self.dyna_v = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

		self.gru = nn.GRU(self.hidden_size*2+self.emb_size,self.hidden_size,batch_first=True)
		self.out = nn.Linear(self.hidden_size,self.decoder_dict_size)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)
		self.log_softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, topic_vector):
		'''input: (batch_size,1,emb_size)
		hidden: (1,batch_size,hidden_size)
		topic_vector: (batch_size,1,hidden_size)'''

		# input_combine: (batch_size,1,hidden_size+emb_size)
		input_combine = torch.cat((input, topic_vector), 2)
		output, hidden = self.gru(input_combine, hidden)

		output = self.log_softmax(self.out(output.squeeze(1)))

		return output, hidden

	def forward(self, input, hidden, list_context_hidden, context_hiddens, topic_vector, pad_matrix):
		'''input: (batch_size,1,emb_size)
		hidden: (1,batch_size,hidden_size)
		topic_vector: (batch_size,1,hidden_size)
		pad_matrix: (batch_size,context_size)'''
		# dyna_attn_weights: (batch_size,context_size)
		dyna_attn_hidden = torch.mm(hidden.squeeze(0),self.dyna_W)
		list_weight = []
		for context_hidden in list_context_hidden:
			# weights:(batch_size,1)
			weight = torch.mm(self.tanh(dyna_attn_hidden+torch.mm(context_hidden.squeeze(0),self.dyna_U)),self.dyna_v)
			list_weight.append(weight)
		# context中padding的句子不参与attention的计算
		dyna_attn_weights = torch.cat(tuple(list_weight),1).masked_fill_(Variable(pad_matrix.cuda()),-100000)
		dyna_attn_weights = self.softmax(dyna_attn_weights)
		
		# dyna_attn_applied: (batch_size,1,hidden_size)
		dyna_attn_applied = torch.bmm(dyna_attn_weights.unsqueeze(1), context_hiddens)

		# input_combine: (batch_size,1,hidden_size*2+emb_size)
		input_combine = torch.cat((input, topic_vector, dyna_attn_applied), 2)
		output, hidden = self.gru(input_combine, hidden)

		output = self.log_softmax(self.out(output.squeeze(1)))

		return output, hidden

class Seq2Seq(nn.Module):
	def __init__(self, dict_size, emb_size, hidden_size, batch_size, dropout, max_senten_len, teach_forcing):
		super(Seq2Seq, self).__init__()
		self.dict_size = dict_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_senten_len = max_senten_len
		self.teach_forcing = teach_forcing

		self.encoder = Encoder(self.emb_size, self.hidden_size)
		self.decoder = Decoder(self.dict_size, self.hidden_size, self.emb_size)
		self.topic_attention = TopicAttention(self.hidden_size)

		# embedding层
		self.embedding = nn.Embedding(self.dict_size,self.emb_size)
		
		'''dropout层: 训练时调用Module.train(),将Dropout的self.training置为True
		预测时调用Module.eval(),将self.training置为False; 或者可以在预测时不使用dropout层'''
		self.dropout = nn.Dropout(self.dropout)

	# 初始化方法
	def init_parameters(self,emb_matrix):
		n = 0
		for name, weight in self.named_parameters():
			if weight.requires_grad:
				#print name,weight.size()
				if  "embedding.weight" in name:
					print (name)
					weight = nn.Parameter(emb_matrix)
				elif weight.ndimension() < 2:
					weight.data.fill_(0)
				else:
					weight.data = init.xavier_uniform(weight.data)
				n += 1
		print ("{} weights requires grad.".format(n))

	def forward(self, reply_tensor_batch, contexts_tensor_batch, pad_matrix_batch, ini_idx):
		'''contexts_tensor_batch数据类型为list,len为max_context_size
		contexts_tensor_batch[i]: batch_size*max_senten_len'''
		encoder_hidden = Variable(torch.zeros(1,self.batch_size,self.hidden_size)).cuda()
		list_context_hidden = []
		list_decoder_output = []

		for context_tensor in contexts_tensor_batch:
			# context_embedded: batch_size*max_senten_len*emb_size
			context_embedded = self.embedding(Variable(context_tensor).cuda())
			context_hidden = self.encoder(self.dropout(context_embedded),encoder_hidden)
			# context_hidden: 1*batch_size*hidden_size
			list_context_hidden.append(context_hidden)

		# attn_weight: batch_size*context_size
		topic_attn_weights = self.topic_attention(list_context_hidden,list_context_hidden[-1],pad_matrix_batch)
		# context_hiddens: batch_size*context_size*hidden_size
		context_hiddens = torch.cat(tuple(list_context_hidden),0).transpose(0,1)
		# topic_vector: batch_size*1*hidden_size
		topic_vector = torch.bmm(topic_attn_weights.unsqueeze(1), context_hiddens)

		decoder_input = torch.cuda.LongTensor([ini_idx]*self.batch_size)
		# context最后一句作为decoder的初始化hidden
		decoder_hidden = self.dropout(list_context_hidden[-1])

		for reply_tensor in reply_tensor_batch:
			decoder_embedded = self.embedding(Variable(decoder_input)).unsqueeze(1)
			decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, list_context_hidden, 
														context_hiddens, topic_vector, pad_matrix_batch)
			list_decoder_output.append(decoder_output)
			if self.teach_forcing:
				# reply_variable：batch_size *
				decoder_input = reply_tensor.cuda()
			else:
				top_values,top_indices = decoder_output.data.topk(1)
				decoder_input = top_indices.squeeze(1)

		return list_decoder_output

	# 预测用函数，输出一个batch对应的预测结果的list
	def predict(self, contexts_tensor_batch, reply_index2word, pad_matrix_batch, ini_idx, sep_char=''):
		encoder_hidden = Variable(torch.zeros(1,self.batch_size,self.hidden_size),volatile=True).cuda()

		list_context_hidden = []
		predict_words_array = [['' for i in range(self.max_senten_len)] for i in range(self.batch_size)]
		predict_sentences = ["" for i in range(self.batch_size)]

		for context_tensor in contexts_tensor_batch:
			context_embedded = self.embedding(Variable(context_tensor,volatile=True).cuda())
			context_hidden = self.encoder(context_embedded,encoder_hidden)
			list_context_hidden.append(context_hidden)

		topic_attn_weights = self.topic_attention(list_context_hidden,list_context_hidden[-1],pad_matrix_batch)
		context_hiddens = torch.cat(tuple(list_context_hidden),0).transpose(0,1)
		topic_vector = torch.bmm(topic_attn_weights.unsqueeze(1), context_hiddens)

		decoder_input = torch.cuda.LongTensor([ini_idx]*self.batch_size)
		decoder_hidden = list_context_hidden[-1]

		for di in range(self.max_senten_len):
			decoder_embedded = self.embedding(Variable(decoder_input,volatile=True)).unsqueeze(1)
			decoder_output,decoder_hidden = self.decoder(decoder_embedded, decoder_hidden, list_context_hidden, 
														context_hiddens, topic_vector, pad_matrix_batch)
			
			decoder_output[:,-2:] = float("-inf")
			top_values,top_indices = decoder_output.data.topk(1)
			batch_topi = [top_indices[i][0] for i in range(self.batch_size)]
			for i in range(self.batch_size):
				predict_words_array[i][di] = reply_index2word[batch_topi[i]]
			decoder_input = top_indices.squeeze(1)

		# 预测的句子以sep_char分隔
		for i in range(self.batch_size):
			predict_sentences[i] = sep_char.join(predict_words_array[i])
		return predict_sentences
