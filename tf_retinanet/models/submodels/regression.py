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

from . import Submodel
from ...losses import smooth_l1
import tensorflow as tf
from ...utils.config import set_defaults


def default_regression_model(
	num_values: int,
	num_anchors: int,
	pyramid_feature_size: int = 256,
	regression_feature_size: int = 256,
	name: str = 'regression_submodel'
):
	""" Creates the default regression submodel.
	Args
		num_values              : Number of values to regress.
		num_anchors             : Number of anchors to regress for each feature level.
		pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
		regression_feature_size : The number of filters to use in the layers in the regression submodel.
		name                    : The name of the submodel.
	Returns
		A tf.keras.models.Model that predicts regression values for each anchor.
	"""
	# All new conv layers except the final one in the
	# RetinaNet (classification) subnets are initialized
	# with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
	options = {
		'kernel_size'        : 3,
		'strides'            : 1,
		'padding'            : 'same',
		'kernel_initializer' : tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
		'bias_initializer'   : 'zeros'
	}

	if tf.keras.backend.image_data_format() == 'channels_first':
		inputs = tf.keras.layers.Input(shape=(pyramid_feature_size, None, None))
	else:
		inputs = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))
	outputs = inputs
	for i in range(4):
		outputs = tf.keras.layers.Conv2D(
			filters=regression_feature_size,
			activation='relu',
			name='pyramid_regression_{}'.format(i),
			**options
		)(outputs)

	outputs = tf.keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
	if tf.keras.backend.image_data_format() == 'channels_first':
		outputs = tf.keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
	outputs = tf.keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

	return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


class BboxRegressionSubmodel(Submodel):
	""" Simple bounding box regression submodel.
	"""
	def __init__(
		self,
		num_values: int,
		num_anchors: int,
		name: str = 'bbox_regression',
	):
		""" Constructor for bbox regression submodel.
		Args
			name       : The name of the submodel.
			num_values : Number of values to regress.
			num_anchors: Number of anchors to regress per feature vector.
		"""
		if num_values < 1:
			raise ValueError(f"expected positive number of values, got {num_values}")

		self.name        = name
		self.num_values  = num_values
		self.num_anchors = num_anchors

		super().__init__()

	def get_name(self):
		""" Return the name of the submodel.
		"""
		return self.name

	def __repr__(self):
		""" Return a description of the model.
		"""
		return 'BboxRegressionSubmodel({})'.format(str(self.num_values))

	def size(self):
		""" Number of regression values.
		"""
		return self.num_values

	def create(self, **kwargs):
		""" Create a regression submodel.
		"""
		return default_regression_model(num_values=self.size(), num_anchors=self.num_anchors, **kwargs)

	def loss(self):
		""" Define a loss function for the regression submodel.
		"""
		return smooth_l1()
