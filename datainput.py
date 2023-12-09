import struct
import numpy as np
from variable import Variable


def decode_mnist_image(mnist_image_file):
	with open(mnist_image_file, 'rb') as f:
		print(f'Solving file {mnist_image_file}')
		fb_data = f.read()

	offset = 0
	fmt_header = '>iiii'  # read 4 uint32 by big-endian
	magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
	print(f'magic number : {magic_number}, num_images: {num_images}')
	offset += struct.calcsize(fmt_header)
	fmt_image = '>' + str(num_rows * num_cols) + 'B'

	images = []
	for i in range(num_images):
		im = struct.unpack_from(fmt_image, fb_data, offset)
		images.append(Variable(np.array(im).reshape((num_rows, num_cols)) / 256))
		offset += struct.calcsize(fmt_image)


	return images


def decode_mnist_label(mnist_label_file):
	with open(mnist_label_file, 'rb') as f:
		print(f'Solving file {mnist_label_file}')
		fb_data = f.read()

	offset = 0
	fmt_header = '>ii'  # read 4 uint32 by big-endian
	magic_number, num_labels = struct.unpack_from(fmt_header, fb_data, offset)
	print(f'magic number : {magic_number}, num_images: {num_labels}')
	offset += struct.calcsize(fmt_header)
	fmt_label = '>' + 'B'

	labels = []
	for i in range(num_labels):
		label = Variable(np.zeros(shape=(10, 1)))
		label.value[struct.unpack_from(fmt_label, fb_data, offset)[0]][0] = 1
		labels.append(label)
		offset += struct.calcsize(fmt_label)

	return labels
