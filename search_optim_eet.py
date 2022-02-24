import pandas as pd
import numpy as np
import os
import re
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# load the trained model
model = keras.models.load_model('../../model-918-tloss-1.474e-07-vloss-1.866e-07.hdf5')
#Show the model architecture
model.summary()


# path to the directory where the training trajectories are stored  
gamma = np.arange(5,70)*5
lamb = np.arange(2,70)*5
temp = np.arange(5,70)*5
num_trajs = len(gamma) 
# Normalize values range of gamma, lambda and temperature ...
# by dividing each value by the maximum value in that range
#
tt_M = 101 #t_M[f]
# Define logistic function    
#
def logistic(x,c):
    a=1
    b=15
    d=1
    f= a/(1 + b * np.exp(-(x-c)/d))
    return f
# We have seven sites and as a label we use ...
# values between 0 and 1
#
states = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#
red_times = 100
t = np.arange(0,1001)*0.005 # 500ps
tt = np.zeros((len(t), red_times), dtype=float)
u=0
for i in t:
    c = -1.0
    for j in range(0, red_times):
        tt[u,j]=logistic(i,c)
        c = c + 5
    u=u + 1
time_steps = np.arange(0,tt.shape[0]) * 5.0
m = 0
if tt_M <= 201:
    for i in range(0,tt_M):
        m = m + 1
if tt_M > 201 and tt_M <= 301:
    for i in range(0,201):
        m = m + 1
    for i in range(202,tt_M, 2):
        m = m + 1
if tt_M > 301 and tt_M <= 501:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,tt_M, 5):
        m = m + 1
if tt_M > 501 and tt_M <= 1001:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,501, 5):
        m = m + 1
    for i in range(510,tt_M, 10):
        m = m + 1
if tt_M > 1001 and tt_M <= 5001:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,501, 5):
        m = m + 1
    for i in range(510,1001, 10):
        m = m + 1
    for i in range(1020,tt_M, 20):
        m = m + 1
if tt_M > 5001 and tt_M <= 10001:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,501, 5):
        m = m + 1
    for i in range(510,1001, 10):
        m = m + 1
    for i in range(1020,5001, 20):
        m = m + 1
    for i in range(5040,tt_M, 40):
        m = m + 1
if tt_M > 10001 and tt_M <= 50001:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,501, 5):
        m = m + 1
    for i in range(510,1001, 10):
        m = m + 1
    for i in range(1020,5001, 20):
        m = m + 1
    for i in range(5040,10001, 40):
        m = m + 1
    for i in range(10100,tt_M, 100):
        m = m + 1
if tt_M > 50001:
    for i in range(0,201):
        m = m + 1
    for i in range(202,301, 2):
        m = m + 1
    for i in range(305,501, 5):
        m = m + 1
    for i in range(510,1001, 10):
        m = m + 1
    for i in range(1020,5001, 20):
        m = m + 1
    for i in range(5040,10001, 40):
        m = m + 1
    for i in range(10100,50001, 100):
        m = m + 1
    for i in range(50200,tt_M, 200):
        m = m + 1
x = np.zeros((m, 5+red_times), dtype=float)
tf = np.zeros((m,1), dtype = float)
#m = m + 1 # include the t -> infinity
# m is the total number of training points 
m = 0
if tt_M < 201:
    for i in range(0,tt_M):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 201 and tt_M <= 301:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,tt_M, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 301 and tt_M <= 501:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,tt_M, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 501 and tt_M <= 1001:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,501, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(510,tt_M, 10):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 1001 and tt_M <= 5001:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,501, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(510,1001, 10):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(1020,tt_M, 20): 
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 5001 and tt_M <= 10001:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,501, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(510,1001, 10):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(1020,5001, 20):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(5040,tt_M, 40):  # 200fs
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 10001 and tt_M <= 50001:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,501, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(510,1001, 10):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(1020,5001, 20):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(5040,10001, 40):  # 200fs
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(10100,tt_M, 100):  #500fs
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
if tt_M > 50001:
    for i in range(0,201):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(202,301, 2):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(305,501, 5):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(510,1001, 10):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(1020,5001, 20):
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(5040,10001, 40):  # 200fs
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(10100,50001, 100):  #500fs
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
    for i in range(50200,tt_M, 200):  # 1ps
        x[m,5:x.shape[1]] = tt[i,:]
        tf[m,0] = time_steps[i]
        m = m + 1
y_pred = np.zeros((1,13), dtype = float)
for f in range(0,2):
    x[:,0] = f
    x[:,1] = 0.3
    if f == 0:
        index = 1
    else:
        index = 6
    for tmp in temp:
        x[:,4] = tmp/310
        for g in gamma:
            x[:,2] = g/300
            for l in lamb:  
                x[:,3] = l/310
                x_pred = x[tt_M-1,:]
                x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
                y_pred[0,:] = model.predict(x_pred, verbose=0)
                parameters = str(index) + "_gamma-" + str(int(g)) + "_lambda-" + str(int(l)) + "_temp-" + str(int(tmp))
                print(parameters, y_pred[0,4])
