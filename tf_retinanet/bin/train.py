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

from ..             import losses
from ..             import models
from ..backbones    import get_backbone
from ..callbacks    import get_callbacks
from ..generators   import get_generators
from ..utils.gpu    import setup_gpu
from ..utils.config import dump_yaml, make_training_config


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--config',    help='Config file.', default=None,        type=str)
	parser.add_argument('--backbone',  help='Backbone model used by retinanet.', type=str)
	parser.add_argument('--generator', help='Generator used by retinanet.',      type=str)

	# Backone config.
	parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
	parser.add_argument('--backbone-weights', help='Weights for the backbone.',           type=str)

	# Generator config.
	parser.add_argument('--random-transform',     help='Randomly transform image and annotations.',                              action='store_true')
	parser.add_argument('--random-visual-effect', help='Randomly visually transform image and annotations.',                     action='store_true')
	parser.add_argument('--batch-size',           help='Size of the batches.',                                                   type=int)
	parser.add_argument('--group-method',         help='Determines how images are grouped together("none", "random", "ratio").', type=str)
	parser.add_argument('--shuffle_groups',       help='If True, shuffles the groups each epoch.',                               action='store_true')
	parser.add_argument('--image-min-side',       help='Rescale the image so the smallest side is min_side.',                    type=int)
	parser.add_argument('--image-max-side',       help='Rescale the image if the largest side is larger than max_side.',         type=int)

	# Train config.
	parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi), -1 to run on cpu.', type=int)
	parser.add_argument('--epochs',           help='Number of epochs to train.',                                          type=int)
	parser.add_argument('--steps',            help='Number of steps per epoch.',                                          type=int)
	parser.add_argument('--lr',               help='Learning rate.',                                                      type=float)
	parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.',                               action='store_true')
	parser.add_argument('--workers',          help='Number of generator workers.',                                        type=int)
	parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.',          type=int)
	parser.add_argument('--weights',          help='Initialize the model with weights from a file.',                      type=str)

	# Additional config.
	parser.add_argument('-o', help='Additional config.',action='append', nargs=1)

	return parser.parse_args(args)


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse command line and configuration file settings.
	config = make_training_config(args)

	# Disable eager, prevents memory leak and makes training faster.
	tf.compat.v1.disable_eager_execution()

	# Set gpu configuration.
	setup_gpu(config['train']['gpu'])

	# Get the submodels manager.
	submodels_manager = models.submodels.SubmodelsManager(config)

	# Get the backbone.
	backbone = get_backbone(config)

	# Get the generators and the submodels updated with info of the generators.
	generators, submodels = get_generators(
		config,
		submodels_manager,
		preprocess_image=backbone.preprocess_image
	)

	if 'train' not in generators:
		raise 'Could not get train generator.'
	train_generator = generators['train']
	validation_generator = None
	if 'validation' in generators:
		validation_generator = generators['validation']
	evaluation_callback = None
	if 'custom_evaluation_callback' in generators:
		evaluation_callback = generators['custom_evaluation_callback']

	# Create the model.
	model = backbone.retinanet(submodels=submodels)

	# If needed load weights.
	if config['train']['weights'] is not None and config['train']['weights'] != 'imagenet':
		model.load_weights(config['train']['weights'], by_name=True)

	# Create prediction model.
	training_model   = model
	prediction_model = models.retinanet.retinanet_bbox(training_model)

	# Create the callbacks.
	callbacks = get_callbacks(
		config['callbacks'],
		model,
		training_model,
		prediction_model,
		validation_generator,
		evaluation_callback,
	)

	# Print model.
	print(training_model.summary())

	loss = {}
	for submodel in submodels:
		loss[submodel.get_name()] = submodel.loss()

	# Compile model.
	training_model.compile(
		loss=loss,
		optimizer=tf.keras.optimizers.Adam(lr=float(config['train']['lr']))
	)

	# Parse training parameters.
	train_config = config['train']

	# Dump the training config in the same folder as the weights.
	dump_yaml(config)

	# Start training.
	return training_model.fit_generator(
		generator=train_generator,
		steps_per_epoch=train_config['steps_per_epoch'],
		epochs=train_config['epochs'],
		verbose=1,
		callbacks=callbacks,
		workers=train_config['workers'],
		use_multiprocessing=train_config['use_multiprocessing'],
		max_queue_size=train_config['max_queue_size'],
	)


if __name__ == '__main__':
	main()
