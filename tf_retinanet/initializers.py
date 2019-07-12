import tensorflow as tf

import numpy as np
import math


class PriorProbability(tf.keras.initializers.Initializer):
	""" Apply a prior probability to the weights.
	"""

	def __init__(self, probability=0.01):
		self.probability = probability

	def get_config(self):
		return {
			'probability': self.probability
		}

	def __call__(self, shape, dtype=None):
		# Set bias to -log((1 - p)/p) for foreground.
		if dtype is not None:
			dtype = dtype.as_numpy_dtype()
		result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

		return result