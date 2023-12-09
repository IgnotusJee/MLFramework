import numpy as np

import variable
from variable import Variable
from typing import List

import datainput
import layer
import lossfunc
import random
import sequential
import trainer


def data_iter_generator(features, labels):
	def inner(batch_size):
		num_data = len(features)
		indices = list(range(num_data))
		random.shuffle(indices)
		for i in range(0, num_data, batch_size):
			batch_indices = [indices[j] for j in range(i, min(i + batch_size, num_data))]
			yield [features[j] for j in batch_indices], [labels[j] for j in batch_indices]
	return inner


batch_size, lr, num_epochs = 256, 0.7, 20

train_iter = data_iter_generator(datainput.decode_mnist_image('./data/MNIST/raw/train-images-idx3-ubyte'),
					   datainput.decode_mnist_label('./data/MNIST/raw/train-labels-idx1-ubyte'))
test_iter = data_iter_generator(datainput.decode_mnist_image('./data/MNIST/raw/t10k-images-idx3-ubyte'),
					  datainput.decode_mnist_label('./data/MNIST/raw/t10k-labels-idx1-ubyte'))

num_input, num_hidden, num_output = 28 * 28, 16, 10


net = sequential.Sequential(
	layer.Flatten(),
	layer.Linear(num_input, num_hidden),
	layer.ReLU(),
	layer.Linear(num_hidden, num_output)
)

loss = lossfunc.CrossEntropyLoss()

updater = trainer.SGD(net, batch_size, lr)


def accuracy(y_hat: List[Variable], y: List[Variable]):
	assert len(y_hat) == len(y)
	delta = 0.
	total = y[0].value.shape[0]
	for yi_hat, yi in zip(y_hat, y):
		res = np.zeros((total, 1))
		s_yi_hat, = layer.Softmax()([yi_hat])
		res[s_yi_hat.value.argmax()][0] = 1
		sumdel = (res == yi.value).all()
		delta += sumdel
	delta /= len(y)
	return delta


def evaluate(net, test_iter, batch_size):
	accu, number = 0, 0
	for X, y in test_iter(batch_size):
		y_hat = net(X)
		accu += accuracy(y_hat, y)
		number += 1
	return accu / number


def train_for_epoch(net: sequential.Sequential, train_iter, loss: lossfunc.LossFunction, updater: trainer.Trainer, batch_size):
	aloss, aaccu, number = 0, 0, 0
	for X, y in train_iter(batch_size):
		updater.zero_grad()
		y_hat = net(X)
		l = loss(y_hat, y)
		l.backward()
		updater.step()
		aloss += l.value.mean()
		aaccu += accuracy(y_hat, y)
		number += 1
	return aloss / number, aaccu / number


def train_model(net: sequential.Sequential, train_iter, test_iter, loss: lossfunc.LossFunction, num_epochs,
				updater: trainer.Trainer, batch_size):
	for epoch in range(num_epochs):
		sloss, saccu = train_for_epoch(net, train_iter, loss, updater, batch_size)
		print(f'{epoch} time, loss : {sloss}, train_acc : {saccu}, test_acc : {evaluate(net, test_iter, batch_size)}')

train_model(net, train_iter, test_iter, loss, num_epochs, updater, batch_size)
