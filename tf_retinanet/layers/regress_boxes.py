import numpy as np
import tensorflow as tf


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

		def _bbox_transform_inv(boxes, deltas, mean=None, std=None):
			""" Applies deltas (usually regression results) to boxes (usually anchors).
			Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
			The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.
			Args
				boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
				deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
				mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
				std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
			Returns
				A np.array of the same shape as boxes, but with deltas applied to each box.
				The mean and std are used during training to normalize the regression values (networks love normalization).
			"""
			if mean is None:
				mean = [0, 0, 0, 0]
			if std is None:
				std = [0.2, 0.2, 0.2, 0.2]

			width  = boxes[:, :, 2] - boxes[:, :, 0]
			height = boxes[:, :, 3] - boxes[:, :, 1]

			x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
			y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
			x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
			y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

			pred_boxes = tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

			return pred_boxes

		return _bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		config = super(RegressBoxes, self).get_config()
		config.update({
			'mean': self.mean.tolist(),
			'std' : self.std.tolist(),
		})

		return config
