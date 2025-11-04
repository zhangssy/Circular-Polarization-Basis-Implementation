import os
import numpy as np
import scipy.io as sio
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat, savemat
import h5py
from time import *
from UNet_11 import *
begin_time = time()
"""# **1.Data processing**"""
def addZeroPadding(X, margin=2):#
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
    ))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX
"""# **2.Dataloader**"""
windowSize =11
num_classes=5
model=UNet().eval().cuda()
model.load_state_dict(torch.load('model path'))
print('model loaded!')
"""# **3.visiualization**"""
T_test=h5py.File('Test dataset','r')
data_hsi1=T1
data_hsi1=torch.from_numpy(data_hsi1.transpose(2,1,0))
height1, width1, c1 = data_hsi1.shape
margin = (windowSize-1)//2
data_hsi1 = addZeroPadding(data_hsi1, margin=margin)
outputs = np.zeros((height1, width1))
for i in range(height1):
    for j in range(width1):
            image_patch = data_hsi1[i:i+windowSize, j:j+windowSize, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).cuda()
            prediction= model(X_test_image).cuda()
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
sio.savemat('results path', {'output': outputs})
print('ALL  Finish!!')
end_time=time()
run_time=end_time-begin_time
h=3600
print('Running time:',run_time/h)