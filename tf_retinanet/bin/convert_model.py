import argparse
import os
import sys

import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin	# noqa: F401
	__package__ = "tf_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..              import models
from ..backbones     import get_backbone
from ..utils.anchors import parse_anchor_parameters
from ..utils.gpu     import setup_gpu
from ..utils.yaml    import parse_yaml


def set_defaults(config):
	# Set defaults for backbone.
	if 'backbone' not in config:
		config['backbone'] = {}
	if 'details' not in config['backbone']:
		config['backbone']['details'] = {}

	# Set defaults for generator.
	if 'generator' not in config:
		config['generator'] = {}
	if 'details' not in config['generator']:
		config['generator']['details'] = {}

	# Set the defaults for convert.
	if 'convert' not in config:
		config['convert'] = {}
	if 'nms' not in config['convert']:
		config['convert']['nms'] = True
	if 'class_specific_filter' not in config['convert']:
		config['convert']['class_specific_filter'] = True
	return config


def parse_args(args):
	parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

	parser.add_argument('model_in',                   help='The model to convert.')
	parser.add_argument('model_out',                  help='Path to save the converted model to.')
	parser.add_argument('--config',                   help='Config file.', default=None, type=str)
	parser.add_argument('--backbone',                 help='The backbone of the model to convert.')
	parser.add_argument('--no-nms',                   help='Disables non maximum suppression.', dest='nms', action='store_false')
	parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')

	return parser.parse_args(args)

def set_args(config, args):
	if args.backbone:
		config['backbone']['name'] = args.backbone

	# Convert config.
	config['convert']['nms'] = args.nms
	config['convert']['class_specific_filter'] = args.class_specific_filter

	return config

def parse_yaml(path):
	with open(path, 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)

def main(args=None, config=None):
	# Parse arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse the configuration file.
	if config is None:
		config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config)

	# Apply the command line arguments to config.
	config = set_args(config, args)

	# Set modified tf session to avoid using the GPUs.
	setup_gpu("cpu")

	# Optionally load anchors parameters.
	anchor_params = None
	if 'anchors' in config['generator']['details']:
		anchor_params = parse_anchor_parameters(config['generator']['details']['anchors'])

	# Get the backbone.
	backbone = get_backbone(config)

	# Load the model.
	model = models.load_model(args.model_in, backbone=backbone)

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
	model.save(args.model_out)


if __name__ == '__main__':
	main()
