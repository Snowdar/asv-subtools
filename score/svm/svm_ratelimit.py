#!/usr/bin/env python

#***************************************
#author: TINA
#date: 2014_10_13 CSLT

# update by Zheng Li at XMUSPEECH
# 2020 07 26

#***************************************

import os, sys, math
import scipy as sp
import numpy as np
from numpy import linalg
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

from sklearn import preprocessing

if __name__ == "__main__":

	fin_train = sys.argv[1]
	fin_test = sys.argv[2]
	fout_ = sys.argv[3]

	maxit = int(sys.argv[4])	
	curve = sys.argv[5]
	Cvalue = float(sys.argv[6])

	sys.stderr.write('Paras: ' + fin_train + ' ' + fin_test + ' ' + fout_ + '\n')

	# data preparation for training set
	
	train_data = []
	train_lable = []

	fin = open(fin_train, 'r')
	sort_ = []
	for line in fin:
		sort_.append(line)
	fin.close()
	sort_.sort()
	fin = open(fin_train, 'w')
	fin.writelines(sort_)
	fin.close()

	fin = open(fin_train, 'r')

	for line in fin:
		line = line.strip()
		wordList = line.split()
		tempList = []
		tempList = np.array(wordList[1:], dtype=float)
		train_data.append(tempList)
		train_lable.append(np.array(wordList[0], dtype=int))

	sys.stderr.write('train_data: ' + str(len(train_data)) + ' ' + 'train_lable: ' + str(len(train_lable)) + '\n')

	fin.close()

	# model training
	#scaler = preprocessing.StandardScaler().fit(train_data)
	#scaler.transform(train_data)

	clf = svm.SVC(C=Cvalue, kernel = curve, probability = True, max_iter = maxit, random_state = 777, class_weight = None, gamma="auto")
	#clf = svm.SVC(kernel = curve,probability = True, max_iter = maxit, random_state = 777, class_weight = 'auto')
	clf.fit(train_data, train_lable)

	sys.stderr.write('Training Done!\n')

	# data preparation for test target test

	test_data = []
	test_lable = []
	
	fin = open(fin_test, 'r')
	sort_ = []
	for line in fin:
		sort_.append(line)
	fin.close()
	sort_.sort()
	fin = open(fin_test, 'w')
	fin.writelines(sort_)
	fin.close()
	
	fin = open(fin_test, 'r')
	
	for line in fin:
		line = line.strip()
		wordList = line.split()
		tempList = []
		tempList = np.array(wordList[1:], dtype=float)
		test_data.append(tempList)
		test_lable.append(np.array(wordList[0], dtype=int))
	sys.stderr.write('test_data: ' + str(len(test_data)) + ' ' + 'test_lable: ' + str(len(test_lable)) + '\n')

	fin.close()	

	#scaler.transform(test_data)

	# predict
	fout = open(fout_, 'w')	
	correct = 0
	incorrect = 0
	for i in range(len(test_data)):
		prob = clf.predict_proba(test_data[i].reshape(1,-1))
		fout.write(str(prob) + '\n')	
	
	fout.close()

#	for i in range(len(test_data)):
#		pre = clf.predict(test_data[i])
#		if pre[0] == test_lable[i]:
#			correct += 1
#		else:
#			incorrect += 1
#	sys.stderr.write('trials: ' + str(len(test_data)) + ' ' + str(correct) + ' ' + str(incorrect))

	print 'Test Done'	
