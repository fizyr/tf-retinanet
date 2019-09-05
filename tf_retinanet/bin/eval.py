import sys
import os
import argparse

import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

from ..              import losses
from ..              import models
from ..backbones     import get_backbone
from ..callbacks     import RedirectModel
from ..generators    import get_generators
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

	# Set defaults for evaluate config.
	if 'evaluate' not in config:
		config['evaluate'] = {}
	if 'convert_model' not in config['evaluate']:
		config['evaluate']['convert_model'] = False
	if 'gpu' not in config['evaluate']:
		config['evaluate']['gpu'] = 0
	if 'score_threshold' not in config['evaluate']:
		config['evaluate']['score_threshold'] = 0.05
	if 'iou_threshold' not in config['evaluate']:
		config['evaluate']['iou_threshold'] = 0.5
	if 'max_detections' not in config['evaluate']:
		config['evaluate']['max_detections'] = 100
	if 'weights' not in config['evaluate']:
		config['evaluate']['weights'] = None

	return config


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--config',    help='Config file.', default=None,        type=str)
	parser.add_argument('--backbone',  help='Backbone model used by retinanet.', type=str)
	parser.add_argument('--generator', help='Generator used by retinanet.',      type=str)

	# Generator config.
	parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',            type=int)
	parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int)

	# Evaluate config.
	parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
	parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi), -1 to run on cpu.',          type=int)
	parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',             type=float)
	parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).',           type=float)
	parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).',                                  type=int)
	parser.add_argument('--weights',         help='Initialize the model with weights from a file.',                               type=str)

	return parser.parse_args(args)


def set_args(config, args):
	if args.backbone:
		config['backbone']['name'] = args.backbone
	if args.generator:
		config['generator']['name'] = args.generator

	# Generator config.
	if args.image_min_side:
		config['generator']['details']['image_min_side'] = args.image_min_side
	if args.image_max_side:
		config['generator']['details']['image_max_side'] = args.image_max_side

	# Evaluate config.
	if args.convert_model:
		config['evaluate']['convert_model'] = args.convert_model
	if args.gpu:
		config['evaluate']['gpu'] = args.gpu
	if args.score_threshold:
		config['evaluate']['score_threshold'] = args.score_threshold
	if args.iou_threshold:
		config['evaluate']['iou_threshold'] = args.iou_threshold
	if args.max_detections:
		config['evaluate']['max_detections'] = args.max_detections
	if args.weights:
		config['evaluate']['weights'] = args.weights

	return config


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse the configuration file.
	config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config)

	# Apply the command line arguments to config.
	config = set_args(config, args)

	# Disable eager, prevents memory leak and makes evaluating faster.
	tf.compat.v1.disable_eager_execution()

	# Set gpu configuration.
	setup_gpu(config['evaluate']['gpu'])

	# Get the backbone.
	backbone = get_backbone(config)

	# Get the generators.
	generators = get_generators(
		config,
		preprocess_image=backbone.preprocess_image
	)

	if 'test' not in generators:
		raise 'Could not get test generator.'
	test_generator = generators['test']
	if 'custom_evaluation' in generators:
		evaluation = generators['custom_evaluation']

	# Load model.
	if config['evaluate']['weights'] is None:
		raise 'Could not get weights.'
	model = models.load_model(config['evaluate']['weights'], backbone=backbone)

	# Create prediction model.
	if config['evaluate']['convert_model']:
		# Optionally load anchors parameters.
		anchor_params = None
		if 'anchors' in config['generator']['details']:
			anchor_params = parse_anchor_parameters(config['generator']['details']['anchors'])

		model = models.retinanet.convert_model(model, anchor_params=anchor_params)

	if not evaluation:
		raise('Standard evaluation not implement yet.')
	evaluation = evaluation(test_generator, model)


if __name__ == '__main__':
	main()
