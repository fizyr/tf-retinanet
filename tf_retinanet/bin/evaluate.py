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

import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

from ..              import models
from ..backbones     import get_backbone
from ..generators    import get_generators
from ..utils.anchors import parse_anchor_parameters
from ..utils.gpu     import setup_gpu
from ..utils.config  import make_evaluation_config


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
	parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi), -1 to run on cpu.', type=int)
	parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',    default=0.05, type=float)
	parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).',  default=0.5,  type=float)
	parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).',                         default=100,  type=int)
	parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).',                  type=str)

	# Additional config.
	parser.add_argument('-o', help='Additional config.',action='append', nargs=1)

	return parser.parse_args(args)


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse command line and configuration file settings.
	config = make_evaluation_config(args)

	# Disable eager, prevents memory leak and makes evaluating faster.
	tf.compat.v1.disable_eager_execution()

	# Set gpu configuration.
	setup_gpu(config['evaluate']['gpu'])

	# Get the submodels manager.
	submodels_manager = models.submodels.SubmodelsManager(config)

	# Get the backbone.
	backbone = get_backbone(config)

	# Get the generators.
	generators, submodels = get_generators(
		config,
		submodels_manager,
		preprocess_image=backbone.preprocess_image
	)

	if 'test' not in generators:
		raise 'Could not get test generator.'
	test_generator = generators['test']

	evaluate = None
	if 'custom_evaluation' not in generators:
		raise('Evaluation not implement yet.')
	if 'custom_evaluation' in generators:
		evaluate = generators['custom_evaluation']

	# Load model.
	if config['evaluate']['weights'] is None:
		raise 'Could not get weights.'
	model = models.load_model(config['evaluate']['weights'], backbone=backbone, submodels=submodels)

	# Create prediction model.
	if config['evaluate']['convert_model']:
		# Optionally load anchors parameters.
		anchor_params = None
		if 'anchors' in config['generator']['details']:
			anchor_params = parse_anchor_parameters(config['generator']['details']['anchors'])

		model = models.retinanet.convert_model(model, anchor_params=anchor_params)

	if config['generator']['name'] == 'coco':
		evaluation = evaluate(test_generator, model)
	else:
		average_precisions, inference_time = evaluate(
			generators['test'],
			model,
			iou_threshold=args.iou_threshold,
			score_threshold=args.score_threshold,
			max_detections=args.max_detections,
			save_path=args.save_path
		)

		# print evaluation
		total_instances = []
		precisions = []
		for label, (average_precision, num_annotations) in average_precisions.items():
			print('{:.0f} instances of class'.format(num_annotations),
				  generators['test'].label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
			total_instances.append(num_annotations)
			precisions.append(average_precision)

		if sum(total_instances) == 0:
			print('No test instances found.')
			return

		print('Inference time for {:.0f} images: {:.4f}'.format(generators['test'].size(), inference_time))
		print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
		print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))


if __name__ == '__main__':
	main()
