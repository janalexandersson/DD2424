import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	with open('Dataset/'+filename, 'rb') as fo:    
		dict = pickle.load(fo, encoding='bytes')
		
		X = np.asarray(dict[b'data'])/255
		
		y = np.asarray(dict[b'labels'])
		
		Y = np.zeros((X.shape[0],10))
		
		Y[np.arange(y.size), y] = 1
		
	return X.transpose(),Y.transpose(), y

def EvaluateClassifier(X, W, b):
	s = np.matmul(W, X) + b
	P = softmax(s)
	return P

	
def ComputeCost(X, Y, W, b, lamb):
	P = EvaluateClassifier(X, W, b)
	J = np.sum(np.diag(-np.log( np.matmul(Y.T, P)))) / X.shape[1] + lamb*np.sum(W**2)
	return J


def ComputeAccuracy(X, y, W, b):
	P = EvaluateClassifier(X, W, b)
	p = np.argmax(P,axis=0).T
	res = p - y
	acc = np.count_nonzero(res==0) / res.shape[0]
	return acc
	
	
def ComputeGradients(X,Y,W,b, lamb):	
	P = EvaluateClassifier(X, W, b)
	g = -(Y-P)
	
	grad_W = np.matmul(g, X.T) / X.shape[1] + 2*lamb*W
	grad_b = np.matmul(g, np.ones((X.shape[1], 1))) / X.shape[1]
	
	return grad_W, grad_b
	


#Taken from the functions.py from the course page
def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

#Taken from the functions.py from the course page
def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]


def RelativeErrorCheck(grad ,ngrad):
	error = np.linalg.norm(ngrad - grad) /  np.maximum(0.000001 , np.linalg.norm(ngrad) + np.linalg.norm(grad))
	return error

def MiniBatch(X_train,Y_train, W, b, n_batch, eta, lamb):
	for j in range(int(X_train.shape[1]/n_batch)):
		j_start = (j-1)*n_batch
		j_end = j*n_batch-1
		X_batch = X_train[:,j_start:j_end]
		Y_batch = Y_train[:,j_start:j_end]
		
		P = EvaluateClassifier(X_batch, W, b)
		grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W,b, lamb)
		W = W - eta*grad_W
		b = b - eta*grad_b
		
	return W, b


def Transform(X, X_train):
	m = (X.transpose() - np.mean(X_train, axis = 1)) / np.std(X_train, axis = 1)
	return m.transpose()

def Train(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')
	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)
	
	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	 
	for epoch in range(n_epochs):
		W,b = MiniBatch(X_train,Y_train, W, b, n_batch, eta, lamb)
		
		cost_train = ComputeCost(X_train, Y_train, W, b, lamb)
		cost_val = ComputeCost(X_val, Y_val, W, b, lamb)
		acc_train = ComputeAccuracy(X_train, y_train, W, b)
		acc_val = ComputeAccuracy(X_val, y_val, W, b)
		
		costs_train.append(cost_train)
		costs_val.append(cost_val)
		
	acc_test = ComputeAccuracy(X_test, y_test, W, b)
	return costs_train, costs_val, W, b, acc_test, acc_train



def plot_cost(costs_train,costs_val):
	epochs_arr = np.arange(0, n_epochs).tolist()

	plt.plot(epochs_arr, costs_train, 'r-',label='Training cost')
	plt.plot(epochs_arr, costs_val, 'b-',label='Validation cost')
	plt.legend(loc='upper right', shadow=True)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()


def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

############################################################################################
#Possible improvements
############################################################################################


def TrainRandomize(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')
	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)

	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	 
	for epoch in range(n_epochs):
		p = np.random.permutation(X_train.shape[0])
		X_train_p = X_train[:,p]
		Y_train_p = Y_train[:,p]
		W,b = MiniBatch(X_train_p,Y_train_p, W, b, n_batch, eta, lamb)
		
	acc_test = ComputeAccuracy(X_test, y_test, W, b)
	return costs_train, costs_val, W, b, acc_test


def TrainDecayingEta(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')
	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)

	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	 
	for epoch in range(n_epochs):
		W,b = MiniBatch(X_train,Y_train, W, b, n_batch, eta, lamb)
		
		
		eta = 0.9*eta
		
	acc_test = ComputeAccuracy(X_test, y_test, W, b)
	return costs_train, costs_val, W, b, acc_test


def TrainLonger(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')

	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)

	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	best_acc_val = 0
	i = 0
	for epoch in range(n_epochs):
		W,b = MiniBatch(X_train,Y_train, W, b, n_batch, eta, lamb)
		
		acc_val = ComputeAccuracy(X_val, y_val, W, b)
		
		if acc_val > best_acc_val:
			best_W = W
			best_b = b
			best_acc_val = acc_val
			i = 0
		
		if i == 50:
			break
		i += 1
		
	best_acc_test = ComputeAccuracy(X_test, y_test, best_W, best_b)
	return costs_train, costs_val, W, b, best_acc_test


def TrainCombined(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')
	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)
	
	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	best_acc_val = 0
	
	i = 0
	for epoch in range(n_epochs):
		p = np.random.permutation(X_train.shape[0])
		X_train_p = X_train[:,p]
		Y_train_p = Y_train[:,p]
		W,b = MiniBatch(X_train_p,Y_train_p, W, b, n_batch, eta, lamb)
		
		acc_val = ComputeAccuracy(X_val, y_val, W, b)
		
		eta = 0.9*eta
		
		if acc_val > best_acc_val:
			best_W = W
			best_b = b
			best_acc_val = acc_val
			i = 0
			
		if i == 50:
			break
		i += 1
		
	best_acc_test = ComputeAccuracy(X_test, y_test, best_W, best_b)
	return costs_train, costs_val, W, b, best_acc_test




#random.seed(1997)
#eta = 0.001
#lamb = 1

#costs_train, costs_val, W, b, acc_test_3, acc_train_3 = Train(n_batch, n_epochs, eta, lamb)
#print("Without improvements: " + str(acc_test_3))

#costs_train, costs_val, W, b, acc_test = TrainDecayingEta(n_batch, n_epochs, eta, lamb)
#print("Decaying Eta: " + str(acc_test))

#costs_train, costs_val, W, b, acc_test = TrainRandomize(n_batch, n_epochs, eta, lamb)
#print("Random perm: " + str(acc_test))

#n_epochs = 1000
#costs_train, costs_val, W, b, acc_test = TrainLonger(n_batch, n_epochs, eta, lamb)
#print("Longer Training: " + str(acc_test))

#costs_train, costs_val, W, b, acc_test = TrainCombined(n_batch, n_epochs, eta, lamb)
#print("Combined: " + str(acc_test))


##########################################################################################
#SVM
##########################################################################################


def ComputeCostSVM(X, Y, W, b, lamb):
	N = X.shape[1]

	s = EvaluateClassifier(X, W, b)
	sc = s.T[np.arange(s.shape[1]), np.argmax(Y, axis=0)].T

	marg = np.maximum(0, s - np.asarray(sc) + 1)
	marg.T[np.arange(N), np.argmax(Y, axis=0)] = 0

	mcsvm_loss = Y.shape[0] * np.mean(np.sum(marg, axis=1))

	cost = 1/N * mcsvm_loss + 0.5 * lamb * np.sum(W**2)

	return cost, marg

#Adjusted from something I found online
#My code was working as intended, however this ran a lot faster so I went with this implementation
def ComputeGradientsSVM(X, Y, W, b, lamb):

	N = X.shape[1]

	_, marg = ComputeCostSVM(X, Y, W, b , lamb)

	bi = marg
	bi[marg > 0] = 1
	bi_sum_rows = np.sum(bi, axis=0)

	bi.T[np.arange(N), np.argmax(Y, axis=0)] = -bi_sum_rows.T

	grad_W = np.dot(bi, X.T) / N + lamb * W

	grad_b = np.reshape(np.sum(bi, axis=1) / bi.shape[1], b.shape)
	return grad_W, grad_b

def MiniBatchSVM(X_train,Y_train, W, b, n_batch, eta, lamb):
	for j in range(int(X_train.shape[1]/n_batch)):
		j_start = (j-1)*n_batch
		j_end = j*n_batch-1
		X_batch = X_train[:,j_start:j_end]
		Y_batch = Y_train[:,j_start:j_end]
		
		P = EvaluateClassifier(X_batch, W, b)
		grad_W, grad_b = ComputeGradientsSVM(X_batch, Y_batch, W, b, lamb)
		W = W - eta*grad_W
		b = b - eta*grad_b
		
	return W, b

def TrainSVM(n_batch, n_epochs, eta, lamb):
	
	X_train,Y_train, y_train = LoadBatch('data_batch_1')
	X_val,Y_val, y_val = LoadBatch('data_batch_2')
	X_test,Y_test, y_test = LoadBatch('test_batch')
	
	X_val = Transform(X_val, X_train)
	X_test = Transform(X_test, X_train)
	X_train = Transform(X_train, X_train)
	
	W = np.random.normal(0,0.01,(Y_train.shape[0], X_train.shape[0]))
	b = np.random.normal(0,0.01,(Y_train.shape[0], 1))
	
	costs_train = []
	costs_val = []
	 
	 
	for epoch in range(n_epochs):
		W,b = MiniBatch(X_train,Y_train, W, b, n_batch, eta, lamb)
		
		cost_train,_ = ComputeCostSVM(X_train, Y_train, W, b, lamb)
		cost_val,_ = ComputeCostSVM(X_val, Y_val, W, b, lamb)
		acc_train = ComputeAccuracy(X_train, y_train, W, b)
		acc_val = ComputeAccuracy(X_val, y_val, W, b)
		
		
		costs_train.append(cost_train)
		costs_val.append(cost_val)
		
	acc_test = ComputeAccuracy(X_test, y_test, W, b)
	return costs_train, costs_val, W, b, acc_test, acc_train

n_batch = 100
n_epochs = 40

random.seed(100)

lamb = 0
eta = 0.1
costs_train, costs_val, W, b, acc_test, acc_train = Train(n_batch, n_epochs, eta, lamb)
print("Cross-ent acc: " + str(acc_test))
costs_train, costs_val, W, b, acc_test, acc_train = TrainSVM(n_batch, n_epochs, eta, lamb)
print("SVM acc: " + str(acc_test))

lamb = 0
eta = 0.001
costs_train, costs_val, W, b, acc_test, acc_train = Train(n_batch, n_epochs, eta, lamb)
print("Cross-ent acc: " + str(acc_test))
costs_train, costs_val, W, b, acc_test, acc_train = TrainSVM(n_batch, n_epochs, eta, lamb)
print("SVM acc: " + str(acc_test))


lamb = 0.1
eta = 0.001
costs_train, costs_val, W, b, acc_test, acc_train = Train(n_batch, n_epochs, eta, lamb)
print("Cross-ent acc: " + str(acc_test))
costs_train, costs_val, W, b, acc_test, acc_train = TrainSVM(n_batch, n_epochs, eta, lamb)
print("SVM acc: " + str(acc_test))


lamb = 1
eta = 0.001
costs_train, costs_val, W, b, acc_test, acc_train = Train(n_batch, n_epochs, eta, lamb)
print("Cross-ent acc: " + str(acc_test))
costs_train, costs_val, W, b, acc_test, acc_train = TrainSVM(n_batch, n_epochs, eta, lamb)
print("SVM acc: " + str(acc_test))

