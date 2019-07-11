import configparser
import numpy as np
import tensorflow as tf
from ..utils.anchors import AnchorParameters


def read_config_file(config_path):
	config = configparser.ConfigParser()

	with open(config_path, 'r') as file:
		config.read_file(file)

	assert 'anchor_parameters' in config, \
		"Malformed config file. Verify that it contains the anchor_parameters section."

	config_keys = set(config['anchor_parameters'])
	default_keys = set(AnchorParameters.default.__dict__.keys())

	assert config_keys <= default_keys, \
		"Malformed config file. These keys are not valid: {}".format(config_keys - default_keys)

	return config


def parse_anchor_parameters(config):
	ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), tf.keras.backend.floatx())
	scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), tf.keras.backend.floatx())
	sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
	strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

	return AnchorParameters(sizes, strides, ratios, scales)
