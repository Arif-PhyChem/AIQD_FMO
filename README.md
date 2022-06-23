# Script for data generation and Code for our article in Nature Communications
# "Predicting the future of excitation energy transfer in light-harvesting complex with artificial intelligence-based quantum dynamics"
 https://www.nature.com/articles/s41467-022-29621-w

1) 	Because of the large size of training data, we couldn't uplaod it here, 
	however you can download quantum_HEOM package from https://github.com/jwa7/quantum_HEOM, 
	and generate the training data. We provide the script LTLME.py

2)	The farthest_point.py samples the trajectories based on farthest point sampling 
	(we just need to do it for one case, initial exciation on site-1 or site-6)

3)	We choose 500 trajectories for site-1 and 500 trajectories from site-6, in total 1000 trajectories as a training set

4)	The validation set is the 100 trajectories from site-1 + 100 trajectories from site-6

5) 	use prep_input.py to prepare your input files accordingly 

6) 	use train_CNN.py to train the CNN model

7)	Run run_dyn.py to predict EET dynamics for test trajectories. We have provided a trained model "trained_ML_model.hdf5" at
	https://doi.org/10.6084/m9.figshare.16922446 (file size exceeds 25MB, GitHub limit)

10)	The respective parameters of test trajectories are in temperature.npy, gamma.npy, initial_site.npy and lambda.npy. 

8) 	The search_optim_eet.py predict population of site-3 at t=0.5ps for different combinations of gamma, lambda and temperature. 
