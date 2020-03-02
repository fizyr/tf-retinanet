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

import cv2


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..                    import models
from ..utils.visualization import draw_annotations, draw_boxes, draw_caption
from ..utils.anchors       import anchors_for_shape, compute_gt_annotations
from ..generators          import get_generators
from ..backbones           import get_backbone
from ..utils.config        import make_debug_config


def parse_args(args):
	""" Parse the arguments.
	"""
	parser = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')

	parser.add_argument('--config', help='Configuration file.',                  type=str, default=None)
	parser.add_argument('--backbone',  help='Backbone model used by retinanet.', type=str)
	parser.add_argument('--generator', help='Generator used by retinanet.',      type=str)

	# Generator config.
	parser.add_argument('--random-transform',     help='Randomly transform image and annotations.',                      action='store_true')
	parser.add_argument('--random-visual-effect', help='Randomly visually transform image and annotations.',             action='store_true')
	parser.add_argument('--image-min-side',       help='Rescale the image so the smallest side is min_side.',            type=int)
	parser.add_argument('--image-max-side',       help='Rescale the image if the largest side is larger than max_side.', type=int)

	# Debug config.
	parser.add_argument('--no-resize',    help='Disable image resizing.',                       action='store_false', dest='resize')
	parser.add_argument('--anchors',      help='Show positive anchors on the image.',           action='store_true')
	parser.add_argument('--display-name', help='Display image name on the bottom left corner.', action='store_true')
	parser.add_argument('--annotations',  help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')

	# Additional config.
	parser.add_argument('-o', help='Additional config.',action='append', nargs=1)

	return parser.parse_args(args)


def run(generator, config):
	""" Main loop.
	Args
		generator : The generator used to debug.
		config    : Dictionary contaning debug configuration.
	"""
	# Display images, one at a time.
	i = 0

	while True:
		# Load the data.
		image       = generator.load_image(i)
		annotations = generator.load_annotations(i)
		if len(annotations['labels']) > 0:
			# Apply random transformations.
			if generator.transform_generator:
				image, annotations, transform = generator.random_transform_group_entry(image, annotations)
			if generator.visual_effect_generator:
				image, annotations = generator.random_visual_effect_group_entry(image, annotations)

			# Resize the image and annotations.
			if config['debug']['resize']:
				image, image_scale = generator.resize_image(image)
				annotations['bboxes'] *= image_scale

			anchors = anchors_for_shape(image.shape)
			positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

			# Draw anchors on the image.
			if config['debug']['anchors']:
				draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

			# Draw annotations on the image.
			if config['debug']['annotations']:
				# Draw annotations in red.
				draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

				# Draw regressed anchors in green to override most red annotations
				# Result is that annotations without anchors are red, with anchors are green.
				draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

			# Display name on the image.
			if config['debug']['display_name']:
				draw_caption(image, [0, image.shape[0]], os.path.basename(generator.image_path(i)))

		cv2.imshow('Image', image)
		key = cv2.waitKey()

		# Use 'a' and 'd' keys for navigation.
		if key == 100:
			i = (i + 1) % generator.size()
		if key == 97:
			i -= 1
			if i < 0:
				i = generator.size() - 1

		# Press q or Esc to quit.
		if (key == ord('q')) or (key == 27):
			return False

	return True


def get_generator(generators):
	""" Retrieve only one of the created generators.
	"""
	if 'validation' in generators:
		return generators['validation']
	if 'test' in generators:
		return generators['test']
	if 'train' in generators:
		return generators['train']

	raise ValueError('Could not call any generator.')


def main(args=None):
	# Parse arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse command line and configuration file settings.
	config = make_debug_config(args)

	# Get the submodels manager.
	submodels_manager = models.submodels.SubmodelsManager(config['submodels'])

	# Get the backbone.
	backbone = get_backbone(config['backbone'])

	# Get generators and submodels.
	generators, submodels = get_generators(
		config['generator'],
		submodels_manager,
		preprocess_image=backbone.preprocess_image
	)

	# Retrieve a single generator.
	generator = get_generator(generators)

	# Create the display window.
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

	# Run the debug.
	run(generator, config)


if __name__ == '__main__':
	main()
