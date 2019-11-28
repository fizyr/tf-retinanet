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
from functools import reduce
import collections.abc

from .defaults import (
		default_training_config,
		default_evaluation_config,
		default_conversion_config
		)


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


def make_conversion_config(args):
	# Parse the configuration file.
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
