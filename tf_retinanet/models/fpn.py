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

from .. import layers


def create_pyramid_features(C3, C4, C5, feature_size=256):
	""" Creates the FPN layers on top of the backbone features.
	Args
		C3           : Feature stage C3 from the backbone.
		C4           : Feature stage C4 from the backbone.
		C5           : Feature stage C5 from the backbone.
		feature_size : The feature size to use for the resulting feature levels.
	Returns
		A list of feature levels [P3, P4, P5, P6, P7].
	"""
	# Upsample C5 to get P5 from the FPN paper.
	P5           = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
	P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
	P5           = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

	# Add P5 elementwise to C4.
	P4           = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
	P4           = tf.keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
	P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
	P4           = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

	# Add P4 elementwise to C3.
	P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
	P3 = tf.keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
	P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

	# "P6 is obtained via a 3x3 stride-2 conv on C5".
	P6 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

	# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6".
	P7 = tf.keras.layers.Activation('relu', name='C6_relu')(P6)
	P7 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

	return [P3, P4, P5, P6, P7]


def build_model_pyramid(name, model, features):
	""" Applies a single submodel to each FPN level.
	Args
		name     : Name of the submodel.
		model    : The submodel to evaluate.
		features : The FPN features.
	Returns
		A tensor containing the response from the submodel on the FPN features.
	"""
	return tf.keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def build_pyramid(models, features):
	""" Applies all submodels to each FPN level.
	Args
		models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
		features : The FPN features.
	Returns
		A list of tensors, one for each submodel.
	"""
	return [build_model_pyramid(n, m, features) for n, m in models]
