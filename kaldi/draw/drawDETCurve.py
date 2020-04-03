# -*- coding:utf-8 -*-

# Copyright xmuspeech （Author:Snowdar 2018-11-29)

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
The two files, trials and scores, need to be provided and this scripy run based on python3.* in windows
 
format of every line in trials_path
	class-id utt-id target/nontarget
	...
	
format of every line in score_path
	class-id utt-id score
	...
	
for suporting kaldi's generation of score
"""
trials_path=sys.argv[1]
score_path=sys.argv[2]
out_png1=sys.argv[3]
out_png2=sys.argv[4]
show=sys.argv[5]


trials=load_data(trials_path,3)
scores=load_data(score_path,3)

dict={}
allScores=[]
farList=[]
frrList=[]
numP=0
numN=0

for x in trials:
	if(x[2]=="target"):
		dict[x[0]+x[1]]=1
	else:
		dict[x[0]+x[1]]=0

for x in scores:
	label=dict[x[0]+x[1]]
	if(label==1):
		numP=numP+1
	else:
		numN=numN+1
		
	allScores.append([float(x[2]),label])


allScores=sorted(allScores,reverse=True)

numFA=0
numFR=numP

eer=50
threshold=0.0
y_max=100
x_max=100 # for drawing with intelligence

farList.append(numFA*1.0/numN*100)	
frrList.append(numFR*1.0/numP*100)

key=0
fixedFar=0.05

for tuple in allScores:
	if(tuple[1]==1):
		numFR=numFR-1
	else:
		numFA=numFA+1
	farList.append(numFA*1.0/numN*100)	
	frrList.append(numFR*1.0/numP*100)
	
	if(numFA*1.0/numN*100>=fixedFar):
		print("FRR%%=%f where FAR%% = 0.05 , threshold = %f"%(numFR*1.0/numP*100,tuple[0]))
		fixedFar=101
		
	# for drawing
	if(numFA*1.0/numN<numFR*1.0/numP):
		eer=numFR*1.0/numP*100
		threshold=tuple[0]
		
	if(numFA==0):
		y_max=numFR*1.0/numP*100
		
	if(numFR!=0):
		x_max=numFA*1.0/numN*100
	if(numFR==0 and key<1):
		if(x_max!=numFA*1.0/numN*100):
			key=key+1
			x_max=numFA*1.0/numN*100
		
	

print("EER%% = %f where threshold = %f"%(eer,threshold))


for i in range(-2,1):
	if(i<0):
		value=np.power(1.0/10,-i)
	else:
		value=np.power(10,i)
	if(eer>value):
		width=value

# draw a complete curve
plt.plot(farList,frrList,linewidth=1,label="EER%%:%f"%(eer))
plt.title("DET Curve")	
plt.xlabel("False Accept probability(%)")
plt.ylabel("False Reject probability(%)")
plt.ylim(0,y_max)
plt.xlim(0,x_max)
plt.grid(True,linestyle='-.')
plt.legend(loc=4)
plt.savefig(out_png1,dpi=256)
if show == 'true' :
	plt.show()

# pay attention to EER part
plt.plot(farList,frrList,linewidth=1,label="EER%%:%f"%(eer))
plt.title("DET Curve")	
plt.xlabel("False Accept probability(%)")
plt.ylabel("False Reject probability(%)")
plt.ylim(0,width*10)
plt.xlim(0,width*10)
plt.grid(True,linestyle='-.')
plt.xticks(np.arange(0, width*10, width)) 
plt.yticks(np.arange(0, width*10, width))
plt.legend(loc=4)
plt.savefig(out_png2,dpi=256)
if show == 'true' :
	plt.show()

