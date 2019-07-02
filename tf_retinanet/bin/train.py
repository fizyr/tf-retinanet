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
from ..backbone import get_backbone
from ..generator import get_generators

def parse_yaml():
	with open("train.yaml", 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)

def main():
	config = parse_yaml()

	backbone = get_backbone(config)
	generators = get_generators(
		config,
		preprocess_image=backbone.preprocess_image
	)

	if not 'train' in generators:
		raise 'Could not get train generator.'
	train_generator = generators['train']

	training_model = backbone.retinanet(train_generator.num_classes())
	prediction_model = models.retinanet.retinanet_bbox(training_model)

	print(training_model.summary())

	# compile model
	training_model.compile(
		loss={
			'regression'    : losses.SmoothL1(),
			'classification': losses.FocalLoss()
		},
		optimizer=tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
	)

	# start training
	return training_model.fit_generator(
		generator=train_generator,
		steps_per_epoch=1000,
		epochs=50,
		verbose=1,
)



if __name__ == '__main__':
	main()
