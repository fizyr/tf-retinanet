class Backbone(object):
	""" This class stores additional information on backbones.
	"""
	def __init__(self, backbone):
		# a dictionary mapping custom layer names to the correct classes
		self.custom_objects = {
			'dummy_custom_a' : 1,
			'dummy_custom_b' : 2,
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
