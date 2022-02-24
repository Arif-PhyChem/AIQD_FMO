import pandas as pd
import numpy as np
import glob
import os
import re

# path to the directory where the training trajectories are stored  
all_files = {}
j = 0
datapath="/mnt/partition-2/data/quantum_HEOM/FMO/fmo_data/training_data"
for files in glob.glob(datapath+'/*.np[yz]'):
    file_name = os.path.basename(files)
    all_files[file_name] = np.load(files)
    j = j + 1
file_count = j
print("number of files = ", file_count)
# create empty list
gamma = np.zeros((file_count), dtype=float)
lamb = np.zeros((file_count), dtype=float)
temp = np.zeros((file_count), dtype=float)
initial = np.zeros((file_count), dtype=int)
j = 0
for files in glob.glob(datapath+'/*.np[yz]'):
    #
    # extract the values of gamma, lambda and temperature from the file name
    #
    file_name = os.path.basename(files)
    x = re.split(r'_', file_name)
    y = re.split(r'-', x[1])
    initial[j] = y[1]
    y = re.split(r'-', x[2]) # extracting value of gamma
    gamma[j] = y[1] 
    y = re.split(r'-', x[3]) # extract value of lambda 
    lamb[j] = y[1]
    y = re.split(r'-', x[4]) 
    x = re.split(r'.npy', y[1]) # extract value of temperature
    temp[j] = x[0]
    j = j + 1
#
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
# Normalize values range of gamma, lambda and temperature ...
# by dividing each value by the maximum value in that range
#
j=0
for i in lamb:
    lamb[j] = i/310
    j=j+1
j=0
for i in gamma:
    gamma[j] = i/300
    j=j+1
j=0
for i in temp:
    temp[j] = i/310
    j=j+1

tmp_M = np.zeros((file_count,len(states)), dtype=int) 
t_M = np.zeros((file_count), dtype=int)
#
# The gradient comparison scheme
#
threshold = 1*10**(-10)
print("threshold = ", threshold)
labels = [1, 9, 17, 25, 33, 41, 49]
i = 0
for files in glob.glob(datapath+'/*.np[yz]'):
    file_name = os.path.basename(files)
    df = all_files[file_name]
    k = 0
    for lab in labels:
        site_data = df[:,lab].real 
        for j in range(0, len(site_data)-10):
            a1 = (site_data[j+1]-site_data[j])/0.005
            a2 = (site_data[j+2]-site_data[j+1])/0.005
            a3 = (site_data[j+3]-site_data[j+2])/0.005
            a4 = (site_data[j+4]-site_data[j+3])/0.005
            a5 = (site_data[j+5]-site_data[j+4])/0.005
            a6 = (site_data[j+6]-site_data[j+5])/0.005
            a7 = (site_data[j+7]-site_data[j+6])/0.005
            a8 = (site_data[j+8]-site_data[j+7])/0.005
            a9 = (site_data[j+9]-site_data[j+8])/0.005
            a10 =(site_data[j+10]-site_data[j+9])/0.005
            if abs(a1) < threshold and abs(a2) < threshold:
                if abs(a3) < threshold and abs(a4) < threshold:
                    if abs(a5) < threshold and abs(a6) < threshold:
                        if abs(a7) < threshold and abs(a8) < threshold:
                            if abs(a9) < threshold and abs(a10) < threshold:
                                tmp_M[i,k] = j   # store the t_M value if all the a1--10 values were less than the threshold
                                k = k  + 1
                                break
    t_M[i] = np.max(tmp_M[i,:])
    i = i + 1
#
# find the total number of training points
#
m = 0
f = 0
for files in glob.glob(datapath+'/*.np[yz]'):
    file_name = os.path.basename(files)
    print(file_name, t_M[f]) # print trajectory and the corresponding time-length for it
    for lab in labels:
        tt_M = t_M[f]
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
    f = f + 1
m = m + (7*file_count)
red_times = 100
# m is the total number of training points 
t = np.arange(0,100001)*0.005 # 500ps
tt = np.zeros((len(t), red_times), dtype=float)
x = np.zeros((m, 5+red_times), dtype=float)
y = np.zeros((m,13), dtype=float)
#
# normalizing the time feature using logistic function 
#
u=0
for i in t:
    c = -1.0
    for j in range(0, red_times):
        tt[u,j]=logistic(i,c)
        c = c + 5
    u=u + 1
m=0
f = 0
for files in glob.glob(datapath+'/*.np[yz]'):
    file_name = os.path.basename(files)
    df = all_files[file_name]
    s = 0
    l = 0
    if initial[f] == 1:   
        init_index = 0  # use 0 as a label for initial excitation on site-1
    else:
        init_index = 1  # use 1 as a label  for initial excitation on site-6
    for lab in range(0, 7):
        site = df[:, 1+l:8+l]
        tt_M = t_M[f]
        if tt_M <= 201:
            for i in range(0,tt_M):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 201 and tt_M <= 301:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1                    
                m = m + 1
            for i in range(202,tt_M, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 301 and tt_M <= 501:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,tt_M, 5):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 501 and tt_M <= 1001:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,501, 5):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(510,tt_M, 10):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 1001 and tt_M <= 5001:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,501, 5):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(510,1001, 10):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(1020,tt_M, 20): 
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 5001 and tt_M <= 10001:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,501, 5):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(510,1001, 10):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(1020,5001, 20): 
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(5040,tt_M, 40):  # 200fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 10001 and tt_M <= 50001:
            for i in range(0,201):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,501, 5):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(510,1001, 10):
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(1020,5001, 20): 
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(5040,10001, 40):  # 200fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(10100,tt_M, 100):  #500fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        if tt_M > 50001:
            for i in range(0,201):  #5fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(202,301, 2):  #10fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(305,501, 5):   #25fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(510,1001, 10):  #50fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(1020,5001, 20):  #100fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(5040,10001, 40):  # 200fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(10100,50001, 100):  #500fs
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
            for i in range(50200,tt_M, 200):  # 1ps
                x[m,0] = init_index
                x[m,1] = states[s]
                x[m,2] = gamma[f]
                x[m,3] = lamb[f]
                x[m,4] = temp[f]
                x[m,5:x.shape[1]] = tt[i,:]
                q = 0
                for p in range(0, 7):
                    if p == s:
                        y[m, q] = site[i,s].real
                        q = q + 1
                    else:
                        y[m, q] = site[i,p].real
                        q = q + 1
                        y[m, q] = site[i,p].imag
                        q = q + 1
                m = m + 1
        x[m,0] = init_index
        x[m,1] = states[s]
        x[m,2] = gamma[f]
        x[m,3] = lamb[f]
        x[m,4] = temp[f]
        x[m,5:x.shape[1]] = 1.0
        q = 0
        for p in range(0, 7):
            if p == s:
                y[m, q] = site[i,s].real
                q = q + 1
            else:
                y[m, q] = site[i,p].real
                q = q + 1
                y[m, q] = site[i,p].imag
                q = q + 1
        l = l + 7
        m = m + 1
        s = s + 1
    f = f + 1
filex = "x.npy"
filey = "y.npy"
np.save(filex, x) # the input is saved in x_10.dat
np.save(filey, y) # the target values are saved in y_10.dat 
