import yaml
import os


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
