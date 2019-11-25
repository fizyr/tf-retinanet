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

import yaml
import os
import operator
from pathlib import Path
from datetime import datetime
from functools import reduce
import collections.abc


def parse_yaml(path):
	with open(path, 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)


def dump_yaml(config):
	print('CONFIG generator: ', config['generator'])
	with open(os.path.join(
		config['callbacks']['snapshots_path'],
		config['callbacks']['project_name'],
		'config.yaml'
	), 'w') as dump_config:
		for key, value in config['generator']['details'].items():
			yaml.dump(value, dump_config, default_flow_style=False)
			print('SUCCESS with key: ', key)


def set_defaults(config, default_config):
	merged_dict = default_config
	for key, value in config.items():
		if isinstance(value, collections.abc.Mapping):
			merged_dict[key] = set_defaults(value, merged_dict.get(key, {}))
		else:
			merged_dict[key] = value

	return merged_dict


def get_drom_dict(datadict, maplist):
	return reduce(operator.getitem, maplist, datadict)


def set_in_dict(datadict, maplist, value):
	get_drom_dict(datadict, maplist[:-1])[maplist[-1]] = value


def parse_additional_options(config, options):
	for option in options:
		split = option[0].split('=')
		value = split[1]
		keys  = split[0].split('.')
		temp_config = config
		set_in_dict(config, keys, value)
	return config


default_backbone_config = {
	'details': {
		'weights': 'imagenet',
		'freeze' : False,
	}
}


default_generator_config = {
	'details': {
		'anchors'                : {},
		'batch_size'             : 1,
		'group_method'           : 'ratio',  # one of 'none', 'random', 'ratio'
		'image_min_side'         : 800,
		'image_max_side'         : 1333,
		'shuffle_groups'         : True,
		'transform_generator'    : None,
		'transform_parameters'   : None,
		'visual_effect_generator': None,
	}
}


default_submodels_config = {
	'retinanet': [
		{'category': 'default_regression',     'name': 'bbox_regression'},
		{'category': 'default_classification', 'name': 'classification'},
	]
}


default_callbacks_config = {
	'snapshots_path': os.path.join(str(Path.home()), 'retinanet-snapshots'),
	'project_name'  : datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
}


default_train_config = {
	'epochs'             : 50,
	'gpu'                : 0,
	'lr'                 : 1e-5,
	'max_queue_size'     : 10,
	'steps_per_epoch'    : 10000,
	'use_multiprocessing': False,
	'weights'            : 'imagenet',
	'workers'            : 1,
}


default_training_config = {
	'backbone' : default_backbone_config,
	'callbacks': default_callbacks_config,
	'generator': default_generator_config,
	'submodels': default_submodels_config,
	'train'    : default_train_config,
}


def make_training_config(args):
	# Parse the configuration file.
	config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config, default_training_config)

	# Additional config; start from this so it can be overwritten by the other command line options.
	if args.o:
		config = parse_additional_options(config, args.o)

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


default_eval_config = {
	'convert_model'  : False,
	'gpu'            : 0,
	'iou_threshold'  : 0.5,
	'max_detections' : 100,
	'score_threshold': 0.05,
	'weights'        : None,
}


default_evaluation_config = {
	'backbone' : default_backbone_config,
	'generator': default_generator_config,
	'submodels': default_submodels_config,
	'evaluate' : default_eval_config,
}


def make_evaluation_config(args):
	# Parse the configuration file.
	config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config, default_evaluation_config)

	# Additional config; start from this so it can be overwritten by the other command line options.
	if args.o:
		config = parse_additional_options(config, args.o)

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

	return config


default_convert_config = {
	'class_specific_filter': True,
	'nms'                  : True,
}


default_conversion_config = {
	'backbone' : default_backbone_config,
	'generator': default_generator_config,
	'submodels': default_submodels_config,
	'convert'  : default_convert_config,
}


def make_conversion_config(args):
	# Parse the configuration file.
	if config is None:
		config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config, default_conversion_config)

	# Additional config; start from this so it can be overwritten by the other command line options.
	if args.o:
		config = parse_additional_options(config, args.o)

	if args.backbone:
		config['backbone']['name'] = args.backbone

	# Convert config.
	config['convert']['nms'] = args.nms
	config['convert']['class_specific_filter'] = args.class_specific_filter

	return config
