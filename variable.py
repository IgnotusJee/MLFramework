import numpy as np

from typing import List, NamedTuple, Callable, Tuple


# record single operation in the order of topo, constructing a calculation graph
class Tape(NamedTuple):
	arguments: List[str]
	values: List[str]
	backward: 'Callable[[List[Variable]], List[Variable]]'


tape_sequence: List[Tape] = []


def reset_tape():
	global _name
	_name = 1
	tape_sequence.clear()


_name = 1


def fresh_name():
	global _name
	name = f'v{_name}'
	_name += 1
	return name


def checker(func):
	def inner(val_a, val_b):
		if not isinstance(val_a, Variable):
			val_a = Variable(val_a)
		if not isinstance(val_b, Variable):
			val_b = Variable(val_b)
		return func(val_a, val_b)

	return inner


class Variable:

	def __init__(self, value, name=None):
		self.value = np.atleast_2d(np.asarray(value))
		self.name = name or fresh_name()

	def __repr__(self):
		return repr(self.value)

	def __str__(self):
		return str(self.value)

	@staticmethod
	def constant(value, name=None):
		var = Variable(value, name)
		# print(f'{var.name} = {value}')
		return var

	def __add__(self, other):
		return ope_add(self, other)

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		return ope_mul(self, other)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		return ope_div(self, other)

	def __rtruediv__(self, other):
		return ope_div(other, self)

	def __sub__(self, other):
		return ope_sub(self, other)

	def __rsub__(self, other):
		return self.__sub__(other)

	def __pow__(self, power):
		return ope_pow(self, power)

	def __matmul__(self, other):
		return ope_matmul(self, other)

	def __neg__(self):
		return ope_neg(self)

	def dimsum(self):
		return ope_dimsum(self)

	def softmax(self):
		return ope_softmax(self)

	def exp(self):
		return ope_exp(self)

	def log(self):
		return ope_log(self)

	@staticmethod
	def max(self, other):
		return ope_max(self, other)

	@staticmethod
	def min(self, other):
		return ope_min(self, other)

	def cross_entropyLoss(self, y):
		return ope_crossentropyLoss(self, y)

	def backward(self):
		return gard(self)

	@staticmethod
	def normal(mean, std, shape: Tuple, name: str = 'normal'):
		return Variable(np.random.normal(mean, std, size=shape), name + fresh_name())

	def shape(self):
		return self.value.shape

	def reshape(self, shape: Tuple):
		return Variable(self.value.reshape(shape))

	@staticmethod
	def zeros(shape: Tuple):
		return Variable(np.zeros(shape))


# operation definition, add closure of backward propagate to the sequence
# Let p of the x be dl/dx
@checker
def ope_add(u1: Variable, u2: Variable):
	v = Variable(u1.value + u2.value, f'({u1.name}+{u2.name})')
	# print(f'operating {u1.value} + {u2.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv, dl_dv]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_mul(u1: Variable, u2: Variable):
	v = Variable(u1.value * u2.value, f'({u1.name}*{u2.name})')
	# print(f'operating {u1.value} * {u2.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		dv_du1 = u2.value
		dv_du2 = u1.value
		return [dl_dv * dv_du1, dl_dv * dv_du2]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_div(u1: Variable, u2: Variable):
	v = Variable(u1.value / u2.value, f'({u1.name}/{u2.name})')
	# print(f'operating {u1.value} / {u2.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		dv_du1 = 1 / u2.value
		dv_du2 = - v.value / u2.value
		return [dl_dv * dv_du1, dl_dv * dv_du2]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_matmul(u1: Variable, u2: Variable):
	v = Variable(u1.value @ u2.value, f'({u1.name}@{u2.name})')
	# print(f'operating {u1.value} @ {u2.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		dv_du1 = u2.value
		dv_du2 = u1.value
		return [dl_dv @ dv_du1.T, dv_du2.T @ dl_dv]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


def ope_neg(u: Variable):
	v = Variable(-u.value, f'neg{u.name}')
	# print(f'operating neg {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [-dl_dv]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v


def ope_dimsum(u: Variable):
	v = Variable(u.value.sum(axis=0).sum(axis=0), f'dimsum{u.name}')
	# print(f'operating dimsum {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * np.ones((1, len(u.value[0])))]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v


def ope_pow(_p: Variable, a: int):
	v = Variable(_p.value ** a, f'({_p.name}**{a})')
	# print(f'operating {_p.value} ** {a} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		dv_du = a * (_p.value ** (a - 1))
		return [dl_dv * dv_du]

	tape_sequence.append(Tape(arguments=[_p.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_sub(u1: Variable, u2: Variable):
	v = Variable(u1.value - u2.value, f'({u1.name}-{u2.name})')
	# print(f'operating {u1.value} - {u2.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv, -1. * dl_dv]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


def ope_softmax(u: Variable):
	e_max = u.value.max() * np.ones_like(u.value)
	e_u = np.exp(u.value - e_max)
	sum_e_u = e_u.sum(axis=0).sum(axis=0)
	v = Variable(e_u / sum_e_u, f'softmax{u.name}')
	# print(f'operating softmax {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * (e_u / sum_e_u - (e_u / sum_e_u) ** 2)]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v


def ope_crossentropyLoss(u: Variable, y: Variable):
	u_max = u.value.max() * np.ones_like(u.value)
	e_u = np.exp(u.value - u_max)
	sum_e_u = e_u.sum(axis=0).sum(axis=0)
	y_hat = e_u / sum_e_u
	log_y_hat = u.value - u_max - np.log(np.exp(u.value - u_max).sum(axis=0).sum(axis=0))

	v = Variable(-(y.value * log_y_hat).sum(axis=0).sum(axis=0), f'CrossEntropyLoss{u.name}')
	# print(f'operating softmax {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * (y_hat - y.value)]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v

def ope_exp(u: Variable):
	v = Variable(np.exp(u.value), f'exp{u.name}')
	# print(f'operating exp {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * v.value]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v


def ope_log(u: Variable):
	v = Variable(np.log(u.value), f'log{u.name}')
	# print(f'operating log {u.value} = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv / u.value]

	tape_sequence.append(Tape(arguments=[u.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_max(u1: Variable, u2: Variable):
	a, b = u1.value, u2.value
	assert a.shape == b.shape
	da, db = (a > b).astype(np.float64), (a < b).astype(np.float64)
	v = Variable(a * da + b * db, f'(max[{u1.name},{u2.name}])')
	# print(f'operating max [{u1.value}, {u2.value}] = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * da, dl_dv * db]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


@checker
def ope_min(u1: Variable, u2: Variable):
	a, b = u1.value, u2.value
	assert a.shape == b.shape
	da, db = (a < b).astype(np.float64), (a > b).astype(np.float64)
	v = Variable(a * da + b * db, f'(min[{u1.name},{u2.name}])')
	# print(f'operating min [{u1.value}, {u2.value}] = {v.value}')

	# guarantee p_v is a 1-dimension vector
	def gradient(p_v):
		assert len(p_v) == 1
		dl_dv, = p_v
		return [dl_dv * da, dl_dv * db]

	tape_sequence.append(Tape(arguments=[u1.name, u2.name], values=[v.name], backward=gradient))
	return v


p = {}


# find all value of the names
def gather_gard(names):
	global p
	return [p[name] if name in p else None for name in names]


# calculating the gard of function l over vector x
def gard(_loss):
	global p
	p = {_loss.name: np.ones_like(_loss.value)}  # map for p of every node indexing by name

	# p[None] = 0.

	# do calc in reversed topo order
	for tape in reversed(tape_sequence):
		# print(tape)
		p_v = gather_gard(tape.values)
		if len(p_v) == 1 and p_v[0] is None:
			continue
		dv_du = tape.backward(p_v)

		for name, value in zip(tape.arguments, dv_du):
			# print(f'{name}: {value}')
			if name in p:
				p[name] += value
			else:
				p[name] = value

	# for name, value in p.items():
	# 	print(f'd({_loss.name})/d({name}) = {value}')


def get_grad(x: List[Variable]):
	return gather_gard(xi.name for xi in x)


def clear_grad():
	reset_tape()
	p.clear()


if __name__ == '__main__':
	x = Variable([[201], [198], [202]], 'x')
	y = Variable([[1], [2], [3]], 'y')
	l = x.softmax()
	print(l)
	l.backward()
	print(get_grad([x]))
