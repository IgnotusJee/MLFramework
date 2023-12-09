from variable import Variable
from typing import List

import datainput
import layer
import lossfunc
import random
import sequential
import trainer


def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	X = [Variable.normal(0, 1, (w.shape()[1], 1)) for _ in range(num_examples)]
	y = [w @ xi + b + Variable.normal(0, 0.01, (1, 1)) for xi in X]
	return X, y


true_w = Variable.constant([2, -3.4])
true_b = Variable.constant(4, 2)
features, labels = synthetic_data(true_w, true_b, 1000)


# print('features:', features[0], '\nlabel:', labels[0])


def data_iter(batch_size, features, labels):
	num_data = len(features)
	indices = list(range(num_data))
	random.shuffle(indices)
	for i in range(0, num_data, batch_size):
		batch_indices = [indices[j] for j in range(i, min(i + batch_size, num_data))]
		yield [features[j] for j in batch_indices], [labels[j] for j in batch_indices]


w = Variable.normal(0, 0.01, shape=(1, 2))
b = Variable.zeros((1, 1))

net = sequential.Sequential(layer.Linear(2, 1, w, b))
loss = lossfunc.SquareLoss()

batch_size, lr = 512, 0.6
updater = trainer.SGD(net, batch_size, lr)
num_epochs = 5

for epoch in range(num_epochs):
	for X, y in data_iter(batch_size, features, labels):
		updater.zero_grad()
		l = loss(net(X), y)
		l.backward()
		updater.step()
	train_l = loss(net(features), labels)
	print(f'epoch {epoch + 1}, loss {train_l}')