"""
Copyright 2017-2019 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf


class ClipBoxes(tf.keras.layers.Layer):
	""" Keras layer to clip box values to lie inside a given shape.
	"""

	def call(self, inputs, **kwargs):
		""" Clips the boxes.
		Args
			inputs : List of [image, boxes] tensors.
		"""
		image, boxes = inputs
		shape = tf.keras.backend.cast(tf.keras.backend.shape(image), tf.keras.backend.floatx())
		if tf.keras.backend.image_data_format() == 'channels_first':
			height = shape[2]
			width  = shape[3]
		else:
			height = shape[1]
			width  = shape[2]
		x1 = tf.clip_by_value(boxes[:, :, 0], 0, width  - 1)
		y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
		x2 = tf.clip_by_value(boxes[:, :, 2], 0, width  - 1)
		y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

		return tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

	def compute_output_shape(self, input_shape):
		""" Computes the output shapes given the input shapes.
		Args
			input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
		Returns
			Tuple representing the output shapes:
		"""
		return input_shape[1]
