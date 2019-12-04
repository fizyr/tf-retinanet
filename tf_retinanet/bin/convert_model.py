#!/usr/bin/env python

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

import argparse
import os
import sys


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..              import models
from ..backbones     import get_backbone
from ..generators    import get_generators
from ..utils.anchors import parse_anchor_parameters
from ..utils.gpu     import setup_gpu
from ..utils.config  import make_conversion_config


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

	parser.add_argument('model_in',                   help='The model to convert.')
	parser.add_argument('model_out',                  help='Path to save the converted model to.')
	parser.add_argument('--config',                   help='Config file.', default=None, type=str)
	parser.add_argument('--backbone',                 help='The backbone of the model to convert.')
	parser.add_argument('--no-nms',                   help='Disables non maximum suppression.',  dest='nms',                   action='store_false')
	parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
	parser.add_argument('--savedmodel',               help='Convert to tensorflow SavedModel.',  dest='savedmodel',            action='store_true')

	# Additional config.
	parser.add_argument('-o', help='Additional config.',action='append', nargs=1)

	return parser.parse_args(args)


def main(args=None, config=None):
	# Parse arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse command line and configuration file settings.
	config = make_conversion_config(args)

	# Set modified tf session to avoid using the GPUs.
	setup_gpu("cpu")

	# Optionally load anchors parameters.
	anchor_params = None
	if 'anchors' in config['generator']['details']:
		anchor_params = parse_anchor_parameters(config['generator']['details']['anchors'])

	# Get the submodels manager.
	submodels_manager = models.submodels.SubmodelsManager(config['submodels'])

	# Get the backbone.
	backbone = get_backbone(config['backbone'])

	# Get the generators and the submodels updated with info of the generators.
	generators, submodels = get_generators(
		config['generator'],
		submodels_manager,
		preprocess_image=backbone.preprocess_image
	)

	# Load the model.
	model = models.load_model(args.model_in, backbone=backbone, submodels=submodels)

	# Check if this is indeed a training model.
	models.retinanet.check_training_model(model)

	# Convert the model.
	model = models.retinanet.convert_model(
		model,
		config['convert']['nms'],
		class_specific_filter=config['convert']['class_specific_filter'],
		anchor_params=anchor_params
	)

	# Save model.
	if not args.savedmodel:
		model.save(args.model_out)
	elif args.savedmodel:
		print('Converting to savedmodel.')
		import tensorflow as tf
		tf.saved_model.save(model, args.model_out)

if __name__ == '__main__':
	main()
