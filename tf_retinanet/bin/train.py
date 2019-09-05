import argparse
import yaml
import sys
import os

import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

from .. import losses
from .. import models
from ..backbones import get_backbone
from ..callbacks import RedirectModel
from ..generators import get_generators
from ..utils.gpu import setup_gpu


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

	# Set defaults for callbacks config.
	if 'callbacks' not in config:
		config['callbacks'] = {}
	if ('snapshots_path' not in config['callbacks']) or (not config['callbacks']['snapshots_path']):
		from pathlib import Path
		home = str(Path.home())
		config['callbacks']['snapshots_path'] = os.path.join(home, 'retinanet-snapshots')
	if ('project_name' not in config['callbacks']) or (not config['callbacks']['project_name']):
		from datetime import datetime
		config['callbacks']['project_name'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

	# Set defaults for train config.
	if 'train' not in config:
		config['train'] = {}
	if 'steps_per_epoch' not in config['train']:
		config['train']['steps_per_epoch'] = 10000
	if 'epochs' not in config['train']:
		config['train']['epochs'] = 50
	if 'use_multiprocessing' not in config['train']:
		config['train']['use_multiprocessing'] = False
	if 'workers' not in config['train']:
		config['train']['workers'] = 1
	if 'max_queue_size' not in config['train']:
		config['train']['max_queue_size'] = 10
	if 'gpu' not in config['train']:
		config['train']['gpu'] = 1
	if 'lr' not in config['train']:
		config['train']['lr'] = 1e-5
	if 'weights' not in config['train']:
		config['train']['weights'] = None
	return config


def parse_yaml(path):
	with open(path, 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)


def create_callbacks(
	config,
	model,
	training_model,
	prediction_model,
	validation_generator=None,
	evaluation_callback=None
):
	callbacks = []

	# Save snapshots of the model.
	os.makedirs(os.path.join(config['snapshots_path'], config['project_name']))
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		os.path.join(
			config['snapshots_path'],
			config['project_name'],
			'{epoch:02d}.h5'
		),
		verbose=1,
	)
	checkpoint = RedirectModel(checkpoint, model)
	callbacks.append(checkpoint)

	# Evaluate the model.
	if validation_generator:
		if not evaluation_callback:
			raise('Standard evaluation_callback not implement yet.')
		evaluation_callback = evaluation_callback(validation_generator)
		evaluation_callback = RedirectModel(evaluation_callback, prediction_model)
		callbacks.append(evaluation_callback)

	return callbacks


def parse_args(args):
	""" Parse the arguments.
	"""
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--config',    help='Config file.', default=None,        type=str)
	parser.add_argument('--backbone',  help='Backbone model used by retinanet.', type=str)
	parser.add_argument('--generator', help='Generator used by retinanet.',      type=str)

	# Backone config.
	parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
	parser.add_argument('--backbone-weights', help='Weights for the backbone.',           type=str)

	# Generator config.
	parser.add_argument('--random-transform',        help='Randomly transform image and annotations.',                   action='store_true')
	parser.add_argument('--random-visual-effect', help='Randomly visually transform image and annotations.',             action='store_true')
	parser.add_argument('--batch-size',       help='Size of the batches.',                                               type=int)
	parser.add_argument('--group-method', help='Determines how images are grouped together("none", "random", "ratio").', type=str)
	parser.add_argument('--shuffle_groups',  help='If True, shuffles the groups each epoch.',                            action='store_true')
	parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.',                type=int)
	parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.',     type=int)

	# Train config.
	parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
	parser.add_argument('--epochs',           help='Number of epochs to train.',                                 type=int)
	parser.add_argument('--steps',            help='Number of steps per epoch.',                                 type=int)
	parser.add_argument('--lr',               help='Learning rate.',                                             type=float)
	parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.',                      action='store_true')
	parser.add_argument('--workers',          help='Number of generator workers.',                               type=int)
	parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int)
	parser.add_argument('--weights',          help='Initialize the model with weights from a file.',             type=str)

	return parser.parse_args(args)


def set_args(config, args):
	if args.backbone:
		config['backbone']['name'] = args.backbone
	if args.generator:
		config['generator']['name'] = args.generator

	# Backbone config.
	if args.freeze_backbone:
		config['backbone']['details']['freeze'] = args.freeze_backbone
	if args.backbone_weights:
		config['backbone']['details']['weights'] = args.backbone_weights

	# Generator config.
	if args.random_transform:
		config['generator']['details']['transform_generator'] = 'random'
	if args.random_visual_effect:
		config['generator']['details']['visual_effect_generator'] = 'random'
	if args.batch_size:
		config['generator']['details']['batch_size'] = args.batch_size
	if args.group_method:
		config['generator']['details']['group_method'] = args.group_method
	if args.shuffle_groups:
		config['generator']['details']['shuffle_groups'] = args.shuffle_groups
	if args.image_min_side:
		config['generator']['details']['image_min_side'] = args.image_min_side
	if args.image_max_side:
		config['generator']['details']['image_max_side'] = args.image_max_side

	# Train config.
	if args.gpu:
		config['train']['gpu'] = args.gpu
	if args.epochs:
		config['train']['epochs'] = args.epochs
	if args.steps:
		config['train']['steps_per_epoch'] = args.steps
	if args.lr:
		config['train']['lr'] = args.lr
	if args.multiprocessing:
		config['train']['use_multiprocessing'] = args.multiprocessing
	if args.workers:
		config['train']['workers'] = args.workers
	if args.max_queue_size:
		config['train']['max_queue_size'] = args.max_queue_size
	if args.weights:
		config['train']['weights'] = args.weights
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

	# Disable eager, prevents memory leak and makes training faster.
	tf.compat.v1.disable_eager_execution()

	# Set gpu configuration.
	setup_gpu(config['train']['gpu'])

	# Get the backbone.
	backbone = get_backbone(config)

	# Get the generators.
	generators = get_generators(
		config,
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
	model = backbone.retinanet(train_generator.num_classes())

	# If needed load weights.
	if config['train']['weights'] is not None and config['train']['weights'] is not 'imagenet':
		model.load_weights(config['train']['weights'], by_name=True)

	# Create prediction model.
	training_model   = model
	prediction_model = models.retinanet.retinanet_bbox(training_model)



	# Create the callbacks.
	callbacks = create_callbacks(
		config['callbacks'],
		model,
		training_model,
		prediction_model,
		validation_generator,
		evaluation_callback,
	)

	# Print model.
	print(training_model.summary())

	# Compile model.
	training_model.compile(
		loss={
			'regression'    : losses.smooth_l1(),
			'classification': losses.focal()
		},
		optimizer=tf.keras.optimizers.Adam(lr=float(config['train']['lr']))
	)

	# Parse training parameters.
	train_config = config['train']

	# Dump the training config in the same folder as the weights.
	with open(os.path.join(
		config['callbacks']['snapshots_path'],
		config['callbacks']['project_name'],
		'config.yaml'
	), 'w') as dump_config:
		yaml.dump(config, dump_config, default_flow_style=False)

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
