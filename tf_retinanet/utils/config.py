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


def parse_yaml(path):
	with open(path, 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			return config
		except yaml.YAMLError as exc:
			raise(exc)


def dump_yaml(config):
	with open(os.path.join(
		config['callbacks']['snapshots_path'],
		config['callbacks']['project_name'],
		'config.yaml'
	), 'w') as dump_config:
		yaml.dump(config, dump_config, default_flow_style=False)


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
