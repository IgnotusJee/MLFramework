from sequential import Sequential
from variable import get_grad, clear_grad


class Trainer:
	seq: Sequential

	def __init__(self, seq: Sequential, batch_size: int):
		self.seq = seq
		self.parameters = seq.parameters()
		self.batch_size = batch_size

	def step(self):
		pass

	def zero_grad(self):
		clear_grad()


class SGD(Trainer):
	def __init__(self, seq: Sequential, batch_size: int, lr):
		super().__init__(seq, batch_size)
		self.lr = lr

	def step(self):
		for layer in self.seq.sequence:
			layer.operate_para(get_grad(layer.parameters()), self.batch_size, self.lr)
