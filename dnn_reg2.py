import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import csv
import sys, os, time, re

from benchmark_funcs import*
from dnn_model import*


def save_scores(best_scores, times, path, filename, nexec):
	output = open(path + filename + "_" + str(nexec) + ".csv", 'w')
	
	wr = csv.writer(output)
	for i in list(range(len(best_scores))):
		wr.writerow([i, best_scores[i], times[i]])

def train_model(X_train, y_train, X_test, y_test, model, nepoch, filename, verbose, nexec, func, layers, neurons, LR, opt, save_log):
	if type(X_train[0]) == type((1,)):
		X = np.array(X_train).reshape(-1,len(X_train[0]),1)
		X_test = np.array(X_test).reshape(-1,len(X_test[0]),1)
	else:
		X = np.array([[a] for a in X_train])
		X_test = np.array([[a] for a in X_test])
		X = np.array(X_train).reshape(-1,1,1)
		X_test = np.array(X_test).reshape(-1,1,1)

	y = [[a] for a in y_train]

	times = []    
	scores = []

	score = 0
	real_init = time.time()
	for i in range(0, nepoch):
		init = time.time()

		model.fit({'input': X}, {'targets': y}, n_epoch=1, show_metric=verbose)
			
		end = time.time()

		y_predicted = model.predict(X_test)
		#print(y_predicted)
		#input()
		score = [abs(abs(y_pred-y_true)/y_true) for y_true, y_pred in zip(y_test, y_predicted)]

		scores.append(float(sum(score)/len(score)))
		times.append(end-init)

	if save_log:
		save_scores(scores, times, "Scores/SCR_", filename, nexec)

	info = open('InfoDNN.csv', 'a')
	if os.stat('InfoDNN.csv').st_size == 0:
		info.write('NEXEC' + ',' + 'FUNC' + ',' + 'OPT' + ',' + 'LR' + ',' + 'HLAYERS' + ',' + 'NEURONS' + ',' + 'MEAN_ERROR' + ',' + 'TRAIN_TIME'+'\n')

	info.write(str(nexec) + ',' + str(func) + ',' + opt + ',' + str(LR) + ',' + str(layers) + ',' + str(neurons) + ',' + str(float(sum(score)/len(score))) + ',' + str(time.time()-real_init) + '\n')

	print('\nError = ' + str(round(float(sum(score)/len(score))*100,2)) + '%')
	return


'''
	Default parameters
'''
LR = 1e-2
nepoch = 1000
verb  = True
fname = "Default_Try"
''''''''''''''''''''''''''''''

'''
	User's parameters (If exists)
'''
for i in range(len(sys.argv)-1):  
	if (sys.argv[i] == '-epoch'):
		nepoch = int(sys.argv[i+1])

	elif(sys.argv[i] == '-LR'):
		LR = float(sys.argv[i+1])    

	elif(sys.argv[i] == '-v'):
		verb = int(sys.argv[i+1])

	elif(sys.argv[i] == '-filename'):
		fname = sys.argv[i+1]                                          



fname = 'DNN_'
n = [5, 10, 100, 1000, 10000]
verb = False

for nexec in range(1,11):
	for opt in ['adam']:
		for LR in [1e-2, 1e-3, 1e-4]:
			for f in [1,2,3,4,5]:
				if f == 1:
					X_train, y_train, X_test, y_test = train_test_f1()
				elif f == 2:
					X_train, y_train, X_test, y_test = train_test_f2()
				elif f == 3:
					X_train, y_train, X_test, y_test = train_test_f3()
				elif f == 4:
					X_train, y_train, X_test, y_test = train_test_f4()
				elif f == 5:
					X_train, y_train, X_test, y_test = train_test_f5()

				for nn in n:
					tf.reset_default_graph()
					if type(X_train[0]) != type((1,)):
						l = 1
					else:
						l = len(X_train[0])

					model = neural_network_model(l, [(nn, 'tanh', -1),(1, 'linear', -1)], opt, LR)
					train_model(X_train = X_train,
								y_train = y_train, 
								X_test  = X_test, 
								y_test  = y_test, 
								model   = model, 
								nepoch  = nepoch, 
								filename = fname + opt + '_F' + str(f) + '_' + str(nn) + '_' + str(LR), 
								verbose = verb, 
								nexec = nexec, 
								func = f, 
								layers = 1, 
								neurons = nn,
								LR = LR,
								opt = opt,
								save_log = True)
