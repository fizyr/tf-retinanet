import yaml
import sys
import os

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

from .. import models
from ..backbone import get_backbone
from ..generator import get_generator

def parse_yaml():
	with open("train.yaml", 'r') as stream:
		try:
			config = yaml.safe_load(stream)
			print(config)
			return config
		except yaml.YAMLError as exc:
			raise(exc)

def main():
	config = parse_yaml()

	backbone = get_backbone(config)
	generator = get_generator(config)

	retinanet = backbone.retinanet(1)
	retinanet = models.retinanet.retinanet_bbox(retinanet)



if __name__ == '__main__':
	main()
