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
