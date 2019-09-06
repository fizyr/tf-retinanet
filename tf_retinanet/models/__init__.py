from . import retinanet  # noqa: F401
import sys

def load_model(filepath, backbone):
	""" Loads a retinanet model using the correct custom objects.
	Args
		filepath: one of the following:
			- string, path to the saved model, or
			- h5py.File object from which to load the model
		backbone: Backbone with which the model was trained.
	Returns
		A tf.keras.models.Model object.
	Raises
		ImportError: if h5py is not available.
		ValueError: In case of an invalid savefile.
	"""
	import tensorflow as tf
	return tf.keras.models.load_model(filepath, custom_objects=backbone.custom_objects)
