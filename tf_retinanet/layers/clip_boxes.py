import tensorflow as tf


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
