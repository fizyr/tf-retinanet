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


def load_model(filepath, backbone, submodels, additional_objects=None):
	""" Loads a retinanet model using the correct custom objects.
	Args
		filepath : one of the following:
			- string, path to the saved model, or
			- h5py.File object from which to load the model
		backbone  : Backbone with which the model was trained.
		submodels : List of submodels used in the model.
	Returns
		A tf.keras.models.Model object.
	Raises
		ImportError : if h5py is not available.
		ValueError  : In case of an invalid savefile.
	"""
	import tensorflow as tf
	custom_objects = backbone.custom_objects
	for submodel in submodels:
		# Get the custom object of the submodel.
		submodel_custom_objects = submodel.get_custom_objects()

		# Check if the custom object is alredy present.
		for key in set(custom_objects).intersection(submodel_custom_objects):
			print('WARNING: custom object present in multiple dicts:', key)

		# Merge the dicts.
		custom_objects.update(submodel_custom_objects)

	# Integrate with additional objects, if provided.
	if additional_objects:
		custom_objects.update(additional_objects)

	return tf.keras.models.load_model(filepath, custom_objects=custom_objects)
