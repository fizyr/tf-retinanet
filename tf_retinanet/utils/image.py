import numpy as np

def preprocess_input(x, mode='caffe'):
	""" Preprocess an image by subtracting the ImageNet mean.
	Args
		x: np.array of shape (None, None, 3) or (3, None, None).
		mode: One of "caffe" or "tf".
			- caffe: will zero-center each color channel with
				respect to the ImageNet dataset, without scaling.
			- tf: will scale pixels between -1 and 1, sample-wise.
	Returns
		The input with the ImageNet mean subtracted.
	"""
	# mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
	# except for converting RGB -> BGR since we assume BGR already

	# covert always to float32 to keep compatibility with opencv
	x = x.astype(np.float32)

	if mode == 'tf':
		x /= 127.5
		x -= 1.
	elif mode == 'caffe':
		x[..., 0] -= 103.939
		x[..., 1] -= 116.779
		x[..., 2] -= 123.68

	return x
