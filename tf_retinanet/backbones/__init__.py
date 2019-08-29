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


def process_backbone_config(config):
	# Get the backbone config.
	backbone_config = config['backbone']['details']

	# If the imagenet weights flag is set for the retinanet model set it for the backbone.
	if 'train' in config and 'weights' in config['train'] and config['train']['weights'] == 'imagenet':
		backbone_config['weights'] = 'imagenet'

	# If the none weights flag is set for the retinanet model set it for the backbone.
	if 'train' in config and 'weights' in config['train'] and config['train']['weights'] == 'none':
		backbone_config['weights'] = 'none'

	# If the weights flag is not set for the backbone set it to imagenet.
	if 'weights' not in backbone_config:
		backbone_config['weights'] = 'imagenet'

	# If the weights flag is set to none pass it to the backbone.
	if backbone_config['weights']  == 'none':
		backbone_config['weights'] = None

	# If freeze flag is not set, set it to false.
	if 'freeze' not in backbone_config:
		backbone_config['freeze'] = False
	return backbone_config


def get_backbone(config):
	try:
		backbone_name = config['backbone']['name']
		backbone_pkg = __import__('tf_retinanet_backbones', fromlist=[backbone_name])
		backbone_pkg = getattr(backbone_pkg, backbone_name)
	except ImportError:
		raise(config['backbone']['name'] + 'is not a valid backbone')

	return backbone_pkg.from_config(process_backbone_config(config))
