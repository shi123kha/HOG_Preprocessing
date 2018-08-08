#!/usr/bin/python
#!/usr/bin/env python -W ignore::DeprecationWarning

import scipy
import numpy as np
import math
import sys
import pickle

rawInputData=sys.argv[1]
inputData=rawInputData.split(",")

#print inputData[len(inputData)-2]

classficationModel=pickle.load(open('imageClassficationModel.sav', 'rb'))
classValue=classficationModel.predict((np.array(inputData[:-1]).reshape(1,-1)).astype(np.float))
print classValue[0]