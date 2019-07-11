class Backbone(object):
	""" This class stores additional information on backbones.
	"""
	def __init__(self, backbone):
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

		self.backbone = backbone
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
	try:
		backbone_name = config['backbone']['name']
		backbone_pkg = __import__('tf_retinanet_backbones', fromlist=[backbone_name])
		backbone_pkg = getattr(backbone_pkg, backbone_name)
	except ImportError:
		raise(config['backbone']['name'] + 'is not a valid backbone')

	return backbone_pkg.from_config(config['backbone']['details'])
