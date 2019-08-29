def freeze(model):
	""" Set all layers in a model to non-trainable.
	The weights for these layers will not be updated during training.
	This function modifies the given model in-place,
	but it also returns the modified model to allow easy chaining with other functions.
	"""
	for layer in model.layers:
		layer.trainable = False
	return model
