# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author:Snowdar 2019-03-17)

# use it with grabLossValue.sh

import sys
import numpy as np
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter
import matplotlib.pyplot as plt

def load_data(data_path,n):
	list=[]
	print("Load data from "+data_path+"...")
	with open(data_path,'r') as f:
		content=f.readlines()
		for line in content:
			line=line.strip()
			data_list=line.split()
			if(n!=len(data_list)):
				print('[error] the %s file has no %s fields'%(data_path,n))
				exit(1)

			list.append(data_list)
	return list

######### main ##########
"""
The format of loss file is as follows:
	index loss-value
"""

dir=sys.argv[1]
out_png=sys.argv[2]
show=sys.argv[3]

loss_path="train.loss diagnostic.loss valid.loss" # train.loss valid.loss"

color=["limegreen","black","red","gold","blue","darkorange","cyan","hotpink","gray",]

loss={}
names=[]
for loss_file in loss_path.split():
	filename=loss_file.split(".")[0]
	names.append(filename)
	loss[filename]=load_data(dir+"/"+loss_file,2)

data={}
for name in names:
	x=[]
	y=[]
	for i in range(0,len(loss[name])):
		x.append(float(loss[name][i][0]))
		y.append(float(loss[name][i][1]))
	
	data[name]=[x,y]

# drawing curve
max_length=0
k=0
for name in names:
	if(len(data[name][0]) > max_length):
		max_length=len(data[name][0])
	plt.plot(data[name][0],data[name][1],linewidth=1,label=name,color=color[k])
	k=(k+1)%len(color)

plt.title("Loss Curve")	
plt.xlabel("Iters")
plt.ylabel("Loss Value")
plt.ylim(-0.5,0.01)
plt.xlim(0,max_length+1)
plt.grid(True,linestyle='-.')
plt.legend(loc=4)
plt.savefig(out_png,256)

if show == 'true' :
	plt.show()

