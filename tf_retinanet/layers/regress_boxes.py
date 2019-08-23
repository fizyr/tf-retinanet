import numpy as np
import tensorflow as tf

from .. import backend


class RegressBoxes(tf.keras.layers.Layer):
	""" Keras layer for applying regression values to boxes.
	"""

	def __init__(self, mean=None, std=None, *args, **kwargs):
		""" Initializer for the RegressBoxes layer.
		Args
			mean: The mean value of the regression values which was used for normalization.
			std: The standard value of the regression values which was used for normalization.
		"""
		if mean is None:
			mean = np.array([0, 0, 0, 0])
		if std is None:
			std = np.array([0.2, 0.2, 0.2, 0.2])

		if isinstance(mean, (list, tuple)):
			mean = np.array(mean)
		elif not isinstance(mean, np.ndarray):
			raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

		if isinstance(std, (list, tuple)):
			std = np.array(std)
		elif not isinstance(std, np.ndarray):
			raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

		self.mean = mean
		self.std  = std
		super(RegressBoxes, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		anchors, regression = inputs
		return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		config = super(RegressBoxes, self).get_config()
		config.update({
			'mean': self.mean.tolist(),
			'std' : self.std.tolist(),
		})

		return config
