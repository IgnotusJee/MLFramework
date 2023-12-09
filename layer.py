import numpy as np
from variable import Variable, fresh_name
from typing import List


class Layer:

	def __call__(self, x: List[Variable]):
		return self.forward(x)

	def forward(self, x: List[Variable]):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError

	# parameters decay
	def parameters(self):
		return []

	def operate_para(self, *args):
		pass


class Linear(Layer):
	w: Variable
	b: Variable
	num_input: int
	num_output: int

	def __init__(self, num_input: int, num_output: int, weight: Variable = None, bias: Variable = None):
		self.num_input = num_input
		self.num_output = num_output
		self.w = Variable(np.random.normal(loc=0, scale=1, size=(num_output, num_input)), 'w' + fresh_name()) if weight is None else weight
		self.b = Variable(np.zeros(shape=(num_output, 1)), 'b' + fresh_name()) if bias is None else bias

	def forward(self, x: List[Variable]):
		for var in x:
			assert var.value.shape == (self.num_input, 1)
		return [self.w @ var + self.b for var in x]

	def parameters(self):
		return [self.w, self.b]

	def operate_para(self, delta, batch_size, lr):
		self.w.value -= delta[0] / batch_size * lr
		self.b.value -= delta[1] / batch_size * lr


class Sigmoid(Layer):

	def forward(self, x: List[Variable]):
		return [1 / (1 + (-var).exp()) for var in x]


class ReLU(Layer):

	def forward(self, x: List[Variable]):
		return [Variable.max(var, np.zeros_like(var.value)) for var in x]


class Tanh(Layer):

	def forward(self, x: List[Variable]):
		return [(1 - (-2 * var).exp()) / (1 + (-2 * var).exp()) for var in x]


class Softmax(Layer):

	def forward(self, x: List[Variable]):
		return [var.softmax() for var in x]


class Flatten(Layer):

	# without grad
	def forward(self, x: List[Variable]):
		length = 1
		for i in x[0].value.shape:
			length *= i
		return [Variable(var.value.flatten(order='C').reshape((length, 1))) for var in x]
