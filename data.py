from os import path
import torch
from torch import tensor
import numpy as np
import string
import linecache


class data:

	# Assume the data is of this form: SpeakerId Text|AddresseeId Text
	
	def __init__(self, params, voc):
		self.params = params
		self.voc = voc
		# EOS: End of source, start of target
		self.EOS = 1
		# EOT: End of target
		self.EOT = 2
		self.padding = 0  # Not used, just a reminder
		self.UNK = params.UNK+params.special_word		#sel.UNK = 3
		
	def encode(self, tokens):
		ids = []
		for token in tokens:
		### For raw-word data:
		# 	try:
		# 		ids.append(self.voc[token]+self.params.special_word)
		# 	except KeyError:
		# 		ids.append(self.UNK)
		###--------------------
		### For data that is already tokenized and transferred to ids:
			# ids.append(int(token)+self.params.special_word)
		### For testing data (numbering starts from 1, not 0):
			ids.append(int(token)-1+self.params.special_word)
		return ids

	def read_batch(self, file, num, mode='train_or_test'):
		origin = []
		sources = np.zeros((self.params.batch_size, self.params.source_max_length+1))	#batch_size*50
		targets = np.zeros((self.params.batch_size, self.params.source_max_length+1))	#batch_size*50
		speaker_label = -np.ones(self.params.batch_size)	#all speaker IDs are set to -1
		addressee_label = -np.ones(self.params.batch_size)
		l_s_set = set()
		l_t_set = set()
		END=0
		a=0
		for i in range(self.params.batch_size):
			if mode == "decode" and self.params.batch_size == 1:
				line = file.strip().split("|")
			else:
				line = linecache.getline(file,num*self.params.batch_size+i+1).strip().split("|")
			i-=a	#to adjust for skipped lines
			if line == ['']:
				END = 1
				break
				
			s = line[-2].split()[:self.params.source_max_length]
			t = line[-1].split()[:self.params.target_max_length]
			
			#skipping lines when Speaker or Addressee speech is empty
			if s[1:]==[]:	#if only one word in Source (i.e Speaker ID)
				a+=1
				continue
			elif t[1:]==[] and mode!='decode':	#if only one word in Target (i.e Addressee ID) AND mode!='decode'
				a+=1
				continue
			
			if self.params.SpeakerMode or self.params.AddresseeMode:
				source=self.encode(s[1:])	#encoding speech of the speaker
				target=[self.EOS]+self.encode(t[1:])+[self.EOT]		#encoding speech of the addressee
			else:
				source=self.encode(s[0:])	#encoding speech of the speaker
				target=[self.EOS]+self.encode(t[0:])+[self.EOT]		#encoding speech of the addressee
			
			l_s=len(source)	#length of Source
			l_t=len(target)	#length of Target
			l_s_set.add(l_s)
			l_t_set.add(l_t)
			
			### If the data contains words, not numbers:
			# origin.append(' '.join(s[1:]))
			origin.append(source)
			sources[i, :l_s]=source		#last few elements will be 0
			targets[i, :l_t]=target		#last few elements will be 0
			try:
				speaker_label[i]=int(s[0])-1	#speaker id (zero-indexed)
				addressee_label[i]=int(t[0])-1	#addressee id (zero-indexed)
			except:
				print('Persona id cannot be transferred to numbers')
			i+=1

		try:
			max_l_s=max(l_s_set)	#length of longest Source sentence in the batch
			max_l_t=max(l_t_set)	#length of longest Target sentence in the batch
		except ValueError:
			return END,None,None,None,None,None,None,None

		if max_l_s == 0:
			return END,None,None,None,None,None,None,None
		elif max_l_t == 2 and mode != 'decode':
			return END,None,None,None,None,None,None,None
			
		
		sources=sources[:i, : max_l_s]		#cutting everything beyong max_l_s
		targets=targets[:i, : max_l_t]		#cutting everything beyong max_l_t
		speaker_label=speaker_label[:i]
		addressee_label=addressee_label[:i]
		
		length_s=(sources!=0).sum(1)	#batch_size, each element is sum of number of words in each sample (includes speaker IDs)
		mask_t=np.ones(targets.shape)*(targets!=0)	# batch_size*max_l_t; 1 in place where the words exist in target, elsewhere 0
		token_num=mask_t[:,1:].sum()	#total number of words in Target for each batch (not including Addressee IDs)

		return END,tensor(sources).long(),tensor(targets).long(),tensor(speaker_label).long(),tensor(addressee_label).long(),tensor(length_s).long(),token_num,origin
