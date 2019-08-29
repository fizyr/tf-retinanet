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


def parse_yaml():
	# TODO get the filename using a parser
	with open(sys.argv[1], 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)


def setup_gpu(gpu_id):
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


def main():
	# Parse the configuration file.
	config = parse_yaml()

	# Disable eager, prevents memory leak and makes training faster.
	tf.compat.v1.disable_eager_execution()

	# Set gpu configuration.
	setup_gpu(0)

	# Get the backbone.
	backbone = get_backbone(config)

	weights = sys.argv[2]

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

	# Create the models.
	model = backbone.retinanet(test_generator.num_classes())
	model.load_weights(weights, by_name=True)

	prediction_model = models.retinanet.retinanet_bbox(model)

	if not evaluation:
		raise('Standard evaluation not implement yet.')
	evaluation = evaluation(test_generator, prediction_model)


if __name__ == '__main__':
	main()
