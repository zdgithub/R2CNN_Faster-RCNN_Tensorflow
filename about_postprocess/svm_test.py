import numpy as np
import scipy.io as sio
import os, sys
from sklearn.externals import joblib
from sklearn import svm as sk_svm
import pickle

# Step 3
datasPath = '../inference_results/testfuv'
datas = os.listdir(datasPath)

clf = joblib.load('trained_theta.pkl')

predPath = '../inference_results/testpred'
if not os.path.exists(predPath):
    os.mkdir(predPath)
    print('testpred is created')


for i in range(len(datas)):
    dataFile = os.path.join(datasPath, datas[i])
    batchx = sio.loadmat(dataFile)['fuv'] # m x 15
    fuv = np.array(batchx)
    pred = clf.decision_function(fuv)
    #print(clf.predict(fuv))
        
    savepath = os.path.join(predPath, datas[i].replace("fuv","pred"))
    sio.savemat(savepath, {'pred' : pred})
