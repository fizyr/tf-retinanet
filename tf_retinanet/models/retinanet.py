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

from ..utils.anchors import AnchorParameters
from .. import initializers
from .. import layers
from . import fpn


def assert_training_model(model):
	""" Assert that the model is a training model.
	"""
	assert(all(output in model.output_names for output in ['bbox_regression', 'classification'])), \
		"Input is not a training model (no 'bbox_regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
	""" Check that model is a training model and exit otherwise.
	"""
	try:
		assert_training_model(model)
	except AssertionError as e:
		import sys
		print(e, file=sys.stderr)
		sys.exit(1)


def build_anchors(anchor_parameters, image, features):
	""" Builds anchors for the shape of the features from FPN.
	Args
		anchor_parameters : Parameteres that determine how anchors are generated.
		image             : The image input tensor.
		features          : The FPN features.
	Returns
		A tensor containing the anchors for the FPN features.
		The shape is:
		```
		(batch_size, num_anchors, 4)
		```
	"""
	anchors = [
		layers.Anchors(
			size=anchor_parameters.sizes[i],
			stride=anchor_parameters.strides[i],
			ratios=anchor_parameters.ratios,
			scales=anchor_parameters.scales,
			name='anchors_{}'.format(i)
		)([image, f]) for i, f in enumerate(features)
	]

	return tf.keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
	inputs,
	backbone_layers,
	submodels,
	num_anchors             = None,
	create_pyramid_features = fpn.create_pyramid_features,
	name                    = 'retinanet'
):
	""" Construct a RetinaNet model on top of a backbone.
	This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
	Args
		inputs                  : keras.layers.Input (or list of) for the input to the model.
		num_anchors             : Number of base anchors.
		create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
		submodels               : Submodels to run on each feature map (default is regression and classification submodels).
		name                    : Name of the model.
	Returns
		A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
		The order of the outputs is as defined in submodels:
		```
		[
			regression, classification, other[0], other[1], ...
		]
		```
	"""

	if num_anchors is None:
		num_anchors = AnchorParameters.default.num_anchors()

	retinanet_submodels = []
	for submodel in submodels:
		retinanet_submodels.append((submodel.get_name(), submodel.create(num_anchors=num_anchors, name='{}_submodel'.format(submodel.get_name()))))

	C3, C4, C5 = backbone_layers

	# Compute pyramid features as per https://arxiv.org/abs/1708.02002.
	features = create_pyramid_features(C3, C4, C5)

	# For all pyramid levels, run available submodels.
	pyramids = fpn.build_pyramid(retinanet_submodels, features)

	return tf.keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
	model                 = None,
	nms                   = True,
	class_specific_filter = True,
	name                  = 'retinanet-bbox',
	anchor_params         = None,
	**kwargs
):
	""" Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

	This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
	These layers include applying the regression values to the anchors and performing NMS.

	Args
		model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
		nms                   : Whether to use non-maximum suppression for the filtering step.
		class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
		name                  : Name of the model.
		anchor_params         : Struct containing anchor parameters. If None, default values are used.
		*kwargs               : Additional kwargs to pass to the minimal retinanet model.

	Returns
		A keras.models.Model which takes an image as input and outputs the detections on the image.

		The order is defined as follows:
		```
		[
			boxes, scores, labels, other[0], other[1], ...
		]
		```
	"""

	# If no anchor parameters are passed, use default values.
	if anchor_params is None:
		anchor_params = AnchorParameters.default

	# Create RetinaNet model.
	if model is None:
		model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
	else:
		assert_training_model(model)

	# Compute the anchors.
	features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
	anchors = build_anchors(anchor_params, model.inputs[0], features)

	# We expect the anchors, regression and classification values as first output.
	regression     = model.outputs[0]
	classification = model.outputs[1]

	# "other" can be any additional output from custom submodels, by default this will be [].
	other = model.outputs[2:]

	# Apply predicted regression to anchors.
	boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
	boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

	# Filter detections (apply NMS / score threshold / select top-k).
	detections = layers.FilterDetections(
		nms                   = nms,
		class_specific_filter = class_specific_filter,
		name                  = 'filtered_detections'
	)([boxes, classification] + other)

	# Construct the model.
	return tf.keras.models.Model(inputs=model.inputs, outputs=detections, name=name)


def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None):
	""" Converts a training model to an inference model.
	Args
		model                 : A retinanet training model.
		nms                   : Boolean, whether to add NMS filtering to the converted model.
		class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
		anchor_params         : Anchor parameters object. If omitted, default values are used.
	Returns
		A tf.keras.models.Model object.
	Raises
		ImportError: if h5py is not available.
		ValueError: In case of an invalid savefile.
	"""
	return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, anchor_params=anchor_params)
