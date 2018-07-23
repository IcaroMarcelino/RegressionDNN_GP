import math
import numpy as np

def f1(x):
	y = 0
	for i in list(range(1, x + 1)):
		y += 1/i
	return y

def f2(x0,x1):
	return 2 - 2.1*math.cos(9.8*x0)*math.sin(1.3*x1)

def f3(x0,x1,x2,x3,x4):
	return 10/(5 + (x0 - 3)**2 + (x1 - 3)**2 + (x2 - 3)**2 + (x3 - 3)**2 + (x4 - 3)**2)

def f4(x):
	return math.log(x + 1) + math.log(1 + x**2)

def f5(x0,x1):
	return (1/(1 + x0**(-4)) + 1/(1 + x1**(-4)))

def train_test_f1():
	X_train = list(range(1, 51))
	X_train = np.random.permutation(X_train)
	y_train = [f1(x) for x in X_train]

	X_test  = list(range(1, 121))
	y_test  = [f1(x) for x in X_test]
	
	return X_train, y_train, X_test, y_test

def train_test_f2():
	x0 = list(np.random.uniform(-50,50,10000))
	x1 = list(np.random.uniform(-50,50,10000))

	X_train = list(zip(x0,x1))
	y_train = [f2(x0,x1) for x0,x1 in X_train]

	x0 = list(np.random.uniform(-50,50,10000))
	x1 = list(np.random.uniform(-50,50,10000))

	X_test  = list(zip(x0,x1))
	y_test  = [f2(x0,x1) for x0,x1 in X_test]
	
	return X_train, y_train, X_test, y_test
	
def train_test_f3():
	x0 = list(np.random.uniform(0.05,6.05,1024))
	x1 = list(np.random.uniform(0.05,6.05,1024))
	x2 = list(np.random.uniform(0.05,6.05,1024))
	x3 = list(np.random.uniform(0.05,6.05,1024))
	x4 = list(np.random.uniform(0.05,6.05,1024))
	
	X_train = list(zip(x0,x1,x2,x3,x4))
	y_train = [f3(x0,x1,x2,x3,x4) for x0,x1,x2,x3,x4 in X_train]

	x0 = list(np.random.uniform(-0.25,6.35,5000))
	x1 = list(np.random.uniform(-0.25,6.35,5000))
	x2 = list(np.random.uniform(-0.25,6.35,5000))
	x3 = list(np.random.uniform(-0.25,6.35,5000))
	x4 = list(np.random.uniform(-0.25,6.35,5000))

	X_test = list(zip(x0,x1,x2,x3,x4))
	y_test  = [f3(x0,x1,x2,x3,x4) for x0,x1,x2,x3,x4 in X_test]
	
	return X_train, y_train, X_test, y_test

def train_test_f4():
	X_train = list(np.random.uniform(0,2,20))
	y_train = [f4(x) for x in X_train]

	X_test  = list(np.random.uniform(0,5,100))
	y_test  = [f4(x) for x in X_test]
	
	return X_train, y_train, X_test, y_test

def train_test_f5():
	x0 = list(range(-50,54,4))
	x0 = [x/10 for x in x0]
	x0 = np.random.permutation(x0)

	x1 = list(range(-50,54,4))
	x1 = [x/10 for x in x1]
	x1 = np.random.permutation(x1)

	X_train = list(zip(x0,x1))
	y_train = [f5(x0,x1) for x0,x1 in X_train]

	x0 = list(range(-100,104,3))
	x0 = [x/10 for x in x0]
	x0 = np.random.permutation(x0)

	x1 = list(range(-100,104,3))
	x1 = [x/10 for x in x1]
	x1 = np.random.permutation(x1)

	X_test  = list(zip(x0,x1))
	y_test  = [f5(x0,x1) for x0,x1 in X_test]
	
	return X_train, y_train, X_test, y_test
