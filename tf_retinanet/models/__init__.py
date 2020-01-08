"""
Copyright 2017-2020 Fizyr (https://fizyr.com)

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

from . import retinanet  # noqa: F401
from . import submodels  # noqa: F401


def load_model(filepath, backbone, submodels, custom_objects=None):
	""" Loads a retinanet model using the correct custom objects.
	Args
		filepath : one of the following:
			- string, path to the saved model, or
			- h5py.File object from which to load the model
		backbone       : Backbone with which the model was trained.
		submodels      : List of submodels used in the model.
		custom_objects : Optional dictionary of custom objects to be passed while loading the model.
	Returns
		A tf.keras.models.Model object.
	Raises
		ImportError : if h5py is not available.
		ValueError  : In case of an invalid savefile.
	"""
	import tensorflow as tf

	if custom_objects is None:
		custom_objects = {}

	# Update custom_objects with the backbone custom objects.
	custom_objects.update(backbone.custom_objects)

	# Update custom_objects with the custom objects of each submodel.
	for submodel in submodels:
		custom_objects.update(submodel.get_custom_objects())

	return tf.keras.models.load_model(filepath, custom_objects=custom_objects)
