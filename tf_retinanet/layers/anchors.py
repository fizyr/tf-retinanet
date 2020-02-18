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
import numpy as np

from ..utils import anchors as utils_anchors
from .. import backend


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

		self.num_anchors = len(self.ratios) * len(self.scales)
		self.anchors     = utils_anchors.generate_anchors(
			base_size=self.size,
			ratios=self.ratios,
			scales=self.scales,
		).astype(np.float32)

		super(Anchors, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		""" Generates the anchors.
		Args
			inputs : List of [image, features] tensors.
		"""
		features_shape  = tf.keras.backend.shape(inputs[1])
		image_shape     = tf.keras.backend.shape(inputs[0])

		# Generate proposals from bbox deltas and shifted anchors.
		if tf.keras.backend.image_data_format() == 'channels_first':
			anchors = backend.shift(image_shape[2:4], features_shape[2:4], self.stride, self.anchors)
		else:
			anchors = backend.shift(image_shape[1:3], features_shape[1:3], self.stride, self.anchors)
		anchors = tf.keras.backend.tile(tf.keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

		return anchors

	def compute_output_shape(self, input_shape):
		""" Computes the output shapes given the input shapes.
		Args
			input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
		Returns
			Tuple representing the output shapes.
		"""
		if None not in input_shape[1][1:]:
			if tf.keras.backend.image_data_format() == 'channels_first':
				total = np.prod(input_shape[1][2:4]) * self.num_anchors
			else:
				total = np.prod(input_shape[1][1:3]) * self.num_anchors

			return (input_shape[1][0], total, 4)
		else:
			return (input_shape[1][0], None, 4)

	def get_config(self):
		""" Gets the configuration of this layer.
		Returns
			Dictionary containing the parameters of this layer.
		"""
		config = super(Anchors, self).get_config()
		config.update({
			'size'   : self.size,
			'stride' : self.stride,
			'ratios' : self.ratios.tolist(),
			'scales' : self.scales.tolist(),
		})

		return config
