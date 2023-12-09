import layer
from variable import Variable
from typing import List


class Sequential:
	sequence: List[layer.Layer]

	def __call__(self, x: Variable):
		next_val = x
		for _layer in self.sequence:
			last_val = next_val
			next_val = _layer.forward(last_val)
		return next_val

	def __init__(self, *layers):
		self.sequence = []
		for _layer in layers:
			self.sequence.append(_layer)

	def parameters(self):
		paras = []
		for _layer in self.sequence:
			paras += _layer.parameters()
		return paras
