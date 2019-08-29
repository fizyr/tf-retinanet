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
from ..utils.version import tf_version_ok


def makedirs(path):  # TODO still needed?
	# Intended behavior: try to create the directory,
	# pass if the directory exists already, fails otherwise.
	# Meant for Python 2.7/3.n compatibility.
	try:
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise


def set_defaults(config):
	if not config['callbacks']['snapshots_path']:
		from pathlib import Path
		home = str(Path.home())
		config['callbacks']['snapshots_path'] = os.path.join(home, 'retinanet-snapshots')
	if not config['callbacks']['project_name']:
		from datetime import datetime
		config['callbacks']['project_name'] = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
	return config


def parse_yaml():
	# TODO get the filename using a parser
	with open(sys.argv[1], 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return set_defaults(config)
		except yaml.YAMLError as exc:
			raise(exc)


def setup_gpu(gpu_id):
	if tf_version_ok((2, 0, 0)):
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			# Restrict TensorFlow to only use the first GPU.
			try:
				# Currently, memory growth needs to be the same across GPUs.
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)

				# Use only the selcted gpu.
				tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
			except RuntimeError as e:
				# Visible devices must be set before GPUs have been initialized.
				print(e)

			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		tf.keras.backend.set_session(tf.Session(config=config))


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
	makedirs(os.path.join(config['snapshots_path'], config['project_name']))
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


def main():
	# Parse the configuration file.
	config = parse_yaml()

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

	# Create the models.
	model            = backbone.retinanet(train_generator.num_classes())
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
		optimizer=tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
	)

	# Parse training parameters.
	train_config = config['train']

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
