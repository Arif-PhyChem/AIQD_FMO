import pandas as pd
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt
import os
import re

#path, dirs, files = next(os.walk("/mnt/partition-2/data/FMO_data/init_1"))
gamma_1 = np.arange(25.0,325.0,25.0) # generate values 25-500 with step-25
lamb_1 = np.arange(10.0,340.0,30.0)
temp_1 = np.arange(30.0,330.0,20.0)
#
file_count = len(gamma_1)*len(lamb_1)*len(temp_1)
print("number of files = ", file_count)
n_traj = file_count;
files = []
for i in range(0,len(gamma_1)):
    for j in range(0,len(lamb_1)):
        for k in range(0,len(temp_1)):
            filename="7_initial-1_wc-" + str(int(gamma_1[i])) + "_lambda-" + str(int(lamb_1[j])) + "_temp-" + str(int(temp_1[k])) + ".npy"
            files.append(filename)
gamma = np.zeros((file_count), dtype=float)
lamb = np.zeros((file_count), dtype=float)
temp = np.zeros((file_count), dtype=float)
initial = np.zeros((file_count), dtype=int)
for i in range(0, file_count):
    # extract the values of gamma, lambda and temperature from the file name
    x = re.split(r'_', files[i])
    y = re.split(r'-', x[1])
    initial[i] = y[1]
    y = re.split(r'-', x[2]) # extracting value of gamma
    gamma[i] = y[1]
    y = re.split(r'-', x[3]) # extract value of lambda
    lamb[i] = y[1]
    y = re.split(r'-', x[4])
    x = re.split(r'.npy', y[1]) # extract value of temperature
    temp[i] = x[0]

# Initialise distances to inf
dists = np.zeros(n_traj, dtype=float) 
dists[:] = float('inf')
points_left = np.arange(n_traj);
sample_inds = np.zeros(n_traj, dtype='int')
# choose an initial trajs
selected = 0
print(files[selected])
#print("7_initial-1_wc-" + str(int(gamma[selected])) + "_lambda-" + str(int(lamb[selected])) + "_temp-" + str(int(temp[selected])) + ".npy")
points_left = np.delete(points_left, selected)
for i in range(1, n_traj):
    last_added = sample_inds[i-1]
    k = 0
    dist_to_last_added_point = np.zeros(len(points_left), dtype=float)
    for j in points_left:
        # Find the distance to the last added traj in selected
        # and all the others
        dist_to_last_added_point[k] =  np.sqrt((gamma[last_added] - gamma[j])**2 + (lamb[last_added] - lamb[j])**2 + (temp[last_added] - temp[j])**2)
        k = k + 1
        # If closer, updated distances
    dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])
    # We want to pick the one that has the largest nearest neighbour
    # distance to the sampled trajectories
    selected = np.argmax(dists[points_left])
    sample_inds[i] = points_left[selected]
    # update the indices of the left trajectories
    print(files[sample_inds[i]])#, (dists[points_left][selected]))
    points_left = np.delete(points_left, selected)


