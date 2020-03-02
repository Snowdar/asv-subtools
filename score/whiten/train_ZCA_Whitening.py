#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: zca.py
# date: Thu May 21 15:47 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""zca: ZCA whitening with a sklearn-like interface
source from https://github.com/mwv/zca
"""

from __future__ import division

import numpy as np
from scipy import linalg
import os,sys
import gc

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

class ZCA(BaseEstimator, TransformerMixin):
	def __init__(self, regularization=1e-6, copy=False):
		self.regularization = regularization
		self.copy = copy

	def fit(self, X, y=None):
		"""Compute the mean, whitening and dewhitening matrices.
		Parameters
		----------
		X : array-like with shape [n_samples, n_features]
			The data used to compute the mean, whitening and dewhitening
			matrices.
		"""
		# X = check_array(X, accept_sparse=None, copy=self.copy,
		#                 ensure_2d=True)
		# if warn_if_not_float(X, estimator=self):
		#     X = X.astype(np.float)
		# self.mean_ = X.mean(axis=0)
		X_ = X #- self.mean_
		cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
		U, S, _ = linalg.svd(cov)
		s = np.sqrt(S.clip(self.regularization))
		s_inv = np.diag(1./s)
		s = np.diag(s)
		self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
		self.dewhiten_ = np.dot(np.dot(U, s), U.T)
		return self

	def transform(self, X, y=None, copy=None):
		"""Perform ZCA whitening
		Parameters
		----------
		X : array-like with shape [n_samples, n_features]
			The data to whiten along the features axis.
		"""
		#check_is_fitted(self, 'mean_')
		X = as_float_array(X, copy=self.copy)
		return np.dot(X , self.whiten_.T)

	def write_matrix(self,filePath):
		with open(filePath,'w') as f:
			f.write(" [")
			for row in self.whiten_:
				f.write("\n ")
				for x in row:
					f.write(" %f"%(x))
				f.write(' 0.0 ') # bias
			f.write("]\n")
		
		
	def inverse_transform(self, X, copy=None):
		"""Undo the ZCA transform and rotate back to the original
		representation
		Parameters
		----------
		X : array-like with shape [n_samples, n_features]
			The data to rotate back.
		"""
		check_is_fitted(self, 'mean_')
		X = as_float_array(X, copy=self.copy)
		return np.dot(X, self.dewhiten_) + self.mean_

	def load_data(self,data_path):
		data=[]
		label=[]
		with open(data_path,'r') as f:
			content=f.readlines()
			for line in content:
				line=line.strip()
				data_list=line.split() 
				if(data_list[1]=='['):
					del data_list[1],data_list[len(data_list)-1]
				spk_id=data_list[0]
				label.append(spk_id)
				del data_list[0]
				vectors= [float(x) for x in data_list]
				data.append(vectors)

		label=np.array(label)
		label=label.reshape(-1,1)

		data=np.array(data)

		return label,data
## class defined end ##
		
def outputdata(label,data_whitened,outFile,ark_format):
	print("Output data to "+outFile+"...")
	label=list(label)
	with open(outFile,'w') as f:
		for i in range(len(data_whitened)):
			if(ark_format):
				f.write(label[i][0]+' [ ')
			else:
				f.write(label[i][0]+' [ ')
			for j in range(len(data_whitened[i])):
				f.write(str(data_whitened[i][j])+' ')
			if(ark_format):
				f.write(']\n')
			else:
				f.write('\n')

def trainMat(train,matrixPath,ark_format=False):
	print("Load train data...")
	trainLabel,trainData=ZCA().load_data(train)
	
	print("data array",trainData.shape)		
	print("Train m matrix by train data...")
	
	trf = ZCA().fit(trainData)
	trf.write_matrix(matrixPath)

# main # xmuspeech ZM 2018-08-23 #
""" Change this script as a called script for kaldi when using ark of text format.
The input file should be this format as follows:
	lable1 6.241 45.3231 -2.352 ...
	lable2 -2.23 15.255 21.5526 ...
	...
or the text format of ark in kaldi 
	lable1 [ 6.241 45.3231 -2.352 ... ]
	lable2 [ -2.23 15.255 21.5526 ... ]
	...
The format of every line is 
	lable-id vector-of-n-dim
or
	lable-id [ vector-of-n-dim ]
The lable-id could be class-id or utt-id or spk_id or any other lable.
The format of output file is as the same as the input file.
"""
options={
"ark_format":False}

n=1
for i in range(1,len(sys.argv)):
    if sys.argv[i].startswith('--'): 
        parameter = sys.argv[i][2:].split("=")
        if(parameter[1]=="true"):
            options[parameter[0].replace("-","_")]=True
        elif(parameter[1]=="false"):
            options[parameter[0].replace("-","_")]=False
        elif(parameter[1]!=""):
            options[parameter[0].replace("-","_")]=parameter[1]
        n+=1
if len(sys.argv)-n != 2 :
	print('usage: '+sys.argv[0]+' [--ark-format=false|true] <train-data> <output-matrix>')
	print('e.g.: '+sys.argv[0]+' train.ark whiten.mat')
	print("note：Reading input file always supports two format,but only when --ark-format=true,then consider the '[' and ']' mark in output file.")
	exit(1)

train=sys.argv[n]
mat=sys.argv[n+1]

sys.stderr.write(sys.argv[0]+': '+train+' '+mat+'\n')
print("Train whitenning mat... ")
trainMat(train,mat,options["ark_format"])
print('Training done!')
