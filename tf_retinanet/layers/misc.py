import tensorflow as tf
import numpy as np

from ..utils import anchors as utils_anchors


class Anchors(tf.keras.layers.Layer):
	""" Keras layer for generating achors for a given shape.
	"""

	def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
		""" Initializer for an Anchors layer.
		Args
			size: The base size of the anchors to generate.
			stride: The stride of the anchors to generate.
			ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
			scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
		"""
		self.size   = size
		self.stride = stride
		self.ratios = ratios
		self.scales = scales

		if ratios is None:
			self.ratios  = utils_anchors.AnchorParameters.default.ratios
		elif isinstance(ratios, list):
			self.ratios  = np.array(ratios)
		if scales is None:
			self.scales  = utils_anchors.AnchorParameters.default.scales
		elif isinstance(scales, list):
			self.scales  = np.array(scales)

		self.num_anchors = len(ratios) * len(scales)
		self.anchors     = tf.keras.backend.variable(utils_anchors.generate_anchors(
			base_size=size,
			ratios=ratios,
			scales=scales,
		))

		super(Anchors, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		features = inputs
		features_shape = tf.keras.backend.shape(features)

		# Generate proposals from bbox deltas and shifted anchors.
		def _shift(shape, stride, anchors):
			""" Produce shifted anchors based on shape of the map and stride size.
			Args
				shape  : Shape to shift the anchors over.
				stride : Stride to shift the anchors with over the shape.
				anchors: The anchors to apply at each location.
			"""
			shift_x = (tf.keras.backend.arange(0, shape[1], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride
			shift_y = (tf.keras.backend.arange(0, shape[0], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride

			shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
			shift_x = tf.keras.backend.reshape(shift_x, [-1])
			shift_y = tf.keras.backend.reshape(shift_y, [-1])

			shifts = tf.keras.backend.stack([
				shift_x,
				shift_y,
				shift_x,
				shift_y
			], axis=0)

			shifts            = tf.keras.backend.transpose(shifts)
			number_of_anchors = tf.keras.backend.shape(anchors)[0]

			k = tf.keras.backend.shape(shifts)[0]  # Number of base points = feat_h * feat_w.

			shifted_anchors = tf.keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + tf.keras.backend.cast(tf.keras.backend.reshape(shifts, [k, 1, 4]), tf.keras.backend.floatx())
			shifted_anchors = tf.keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

			return shifted_anchors

		if tf.keras.backend.image_data_format() == 'channels_first':
			anchors = _shift(features_shape[2:4], self.stride, self.anchors)
		else:
			anchors = _shift(features_shape[1:3], self.stride, self.anchors)
		anchors = tf.keras.backend.tile(tf.keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

		return anchors

	def compute_output_shape(self, input_shape):
		if None not in input_shape[1:]:
			if tf.keras.backend.image_data_format() == 'channels_first':
				total = np.prod(input_shape[2:4]) * self.num_anchors
			else:
				total = np.prod(input_shape[1:3]) * self.num_anchors

			return (input_shape[0], total, 4)
		else:
			return (input_shape[0], None, 4)

	def get_config(self):
		config = super(Anchors, self).get_config()
		config.update({
			'size'   : self.size,
			'stride' : self.stride,
			'ratios' : self.ratios.tolist(),
			'scales' : self.scales.tolist(),
		})

		return config


class UpsampleLike(tf.keras.layers.Layer):
	""" Keras layer for upsampling a Tensor to be the same shape as another Tensor.
	"""

	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = tf.keras.backend.shape(target)
		if tf.keras.backend.image_data_format() == 'channels_first':
			source = tf.transpose(source, (0, 2, 3, 1))
			output = tf.image.resize(source, (target_shape[2], target_shape[3]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			output = tf.transpose(output, (0, 3, 1, 2))
			return output
		else:
			return tf.image.resize(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	def compute_output_shape(self, input_shape):
		if tf.keras.backend.image_data_format() == 'channels_first':
			return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
		else:
			return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


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


class ClipBoxes(tf.keras.layers.Layer):
	""" Keras layer to clip box values to lie inside a given shape.
	"""

	def call(self, inputs, **kwargs):
		image, boxes = inputs
		shape = tf.keras.backend.cast(tf.keras.backend.shape(image), tf.keras.backend.floatx())
		if tf.keras.backend.image_data_format() == 'channels_first':
			height = shape[2]
			width  = shape[3]
		else:
			height = shape[1]
			width  = shape[2]
		x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
		y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
		x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
		y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

		return tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

	def compute_output_shape(self, input_shape):
		return input_shape[1]
