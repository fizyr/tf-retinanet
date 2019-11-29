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

from ..utils import import_package


class Backbone(object):
	""" This class stores additional information on backbones.
	"""
	def __init__(self, config):
		# a dictionary mapping custom layer names to the correct classes
		from .. import layers
		from .. import losses
		from .. import initializers
		self.custom_objects = {
			'UpsampleLike'     : layers.UpsampleLike,
			'PriorProbability' : initializers.PriorProbability,
			'RegressBoxes'     : layers.RegressBoxes,
			'FilterDetections' : layers.FilterDetections,
			'Anchors'          : layers.Anchors,
			'ClipBoxes'        : layers.ClipBoxes,
			'_smooth_l1'       : losses.smooth_l1(),
			'_focal'           : losses.focal(),
		}

		self.backbone = config['type']
		self.weights  = config['weights']
		self.modifier = None
		if config['freeze']:
			from ..utils.model import freeze as freeze_model
			self.modifier = freeze_model
		self.validate()

	def retinanet(self, *args, **kwargs):
		""" Returns a retinanet model using the correct backbone.
		"""
		raise NotImplementedError('retinanet method not implemented.')

	def validate(self):
		""" Checks whether the backbone string is correct.
		"""
		raise NotImplementedError('validate method not implemented.')

	def preprocess_image(self, inputs):
		""" Takes as input an image and prepares it for being passed through the network.
		Having this function in Backbone allows other backbones to define a specific preprocessing step.
		"""
		raise NotImplementedError('preprocess_image method not implemented.')


def get_backbone(config):
	backbone_pkg = import_package(config['backbone']['name'], 'tf_retinanet_backbones')

	return backbone_pkg.from_config(config['backbone']['details'])
