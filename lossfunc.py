from variable import Variable, fresh_name
from typing import List
import numpy as np


class LossFunction:
	def __call__(self, y_hat: List[Variable], y: List[Variable]):
		pass


class SquareLoss(LossFunction):
	def __call__(self, y_hat: List[Variable], y: List[Variable]):
		_loss = Variable(np.zeros((1, 1)), 'SquareLoss' + fresh_name())
		for yi_hat, yi in zip(y_hat, y):
			_loss += (yi_hat - yi) ** 2 / 2
		return _loss


class CrossEntropyLoss(LossFunction):
	def __call__(self, o: List[Variable], y: List[Variable]):
		_loss = Variable(np.zeros((1, 1)), 'CrossEntropy' + fresh_name())
		for oi, yi in zip(o, y):
			_loss += oi.cross_entropyLoss(yi)
		return _loss


class CrossEnT(LossFunction):
	def __call__(self, y_hat: List[Variable], y: List[Variable]):
		_loss = Variable(np.zeros((1, 1)), 'CrossEnT' + fresh_name())
		for yi_hat, yi in zip(y_hat, y):
			_loss += - (yi * Variable.log(yi_hat)).dimsum()
		return _loss