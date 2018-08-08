#!/usr/bin/python
#!/usr/bin/env python -W ignore::DeprecationWarning
import scipy
import numpy as np
import math
import sys
import pickle
from sklearn.svm import LinearSVC
from sklearn import svm

def readData(fileName):
	fin=open(fileName,"r")
	
	data=[]
	time=[]
	trainData=[]
	label=[]
	
	for line in fin:
		tempSplit=line[:-2].split(",")
		#print tempSplit,
		trainData.append(tempSplit[:-1])
		label.append(tempSplit[len(tempSplit)-1])

	print label
	data.append(trainData)
	data.append(label)
	return data
	

print "File Name=",sys.argv[1]
inputData=readData(sys.argv[1])

print "Data read complted"

#Linear model for train
clf = svm.LinearSVC()
clf.fit(np.array(inputData[0]).astype(np.float),np.array(inputData[1]).astype(np.float))

#Saving Model

pickle.dump(clf, open('imageClassficationModel.sav', 'wb'))
print "Successful train"