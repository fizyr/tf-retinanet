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


class UpsampleLike(tf.keras.layers.Layer):
	""" Keras layer for upsampling a Tensor to be the same shape as another Tensor.
	"""

	def call(self, inputs, **kwargs):
		""" Upsamples the tensor.
		Args
			inputs : List of [source, target] tensors.
		"""
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
		""" Computes the output shapes given the input shapes.
		Args
			input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
		Returns
			Tuple representing the output shapes.
		"""
		if tf.keras.backend.image_data_format() == 'channels_first':
			return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
		else:
			return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
