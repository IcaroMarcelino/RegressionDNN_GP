import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def neural_network_model(input_size, architecture, optimizer, LR):
	network = input_data(shape=[None, input_size, 1], name='input')

	for neurons, activation, drop_out in architecture:
		network = fully_connected(network, neurons, activation=activation)

		if drop_out != -1:
			network = dropout(network, drop_out)

	network = regression(network, optimizer=optimizer, learning_rate=LR, loss='mean_square', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model