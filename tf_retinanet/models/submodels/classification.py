from . import Submodel
from ... import initializers
from ...losses import focal
import tensorflow as tf


def default_classification_model(
	num_classes,
	num_anchors,
	pyramid_feature_size=256,
	prior_probability=0.01,
	classification_feature_size=256,
	name='classification_submodel'
):
	""" Creates the default regression submodel.
	Args
		num_classes                 : Number of classes to predict a score for at each feature level.
		num_anchors                 : Number of anchors to predict classification scores for at each feature level.
		pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
		prior_probability           : Probability for the bias initializer of the last convolutional layer.
		classification_feature_size : The number of filters to use in the layers in the classification submodel.
		name                        : The name of the submodel.
	Returns
		A tensorflow.keras.models.Model that predicts classes for each anchor.
	"""
	options = {
		'kernel_size' : 3,
		'strides'     : 1,
		'padding'     : 'same',
	}

	if tf.keras.backend.image_data_format() == 'channels_first':
		inputs = tf.keras.layers.Input(shape=(pyramid_feature_size, None, None))
	else:
		inputs = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))
	outputs = inputs
	for i in range(4):
		outputs = tf.keras.layers.Conv2D(
			filters=classification_feature_size,
			activation='relu',
			name='pyramid_classification_{}'.format(i),
			kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
			bias_initializer='zeros',
			**options
		)(outputs)

	outputs = tf.keras.layers.Conv2D(
		filters=num_classes * num_anchors,
		kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
		bias_initializer=initializers.PriorProbability(probability=prior_probability),
		name='pyramid_classification',
		**options
	)(outputs)

	# Reshape output and apply sigmoid.
	if tf.keras.backend.image_data_format() == 'channels_first':
		outputs = tf.keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
	outputs = tf.keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
	outputs = tf.keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

	return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


class ClassificationSubmodel(Submodel):
	""" simple classification submodel, performing multi-class prediction.
	"""
	def __init__(self, config, **kwargs):
		""" Constructor for "standard" classification submodel.
		Args
			config: Defines the configuration for the submodel.
			TODO: list required info in config
		"""
		assert('num_classes' in config), "Number of classes not in config."

		if 'name' not in config:
			config['name'] = 'classification'
		self.name = config['name']

		self.num_classes = config['num_classes']
		assert(self.num_classes > 0), "No classes to classify."

		super(ClassificationSubmodel, self).__init__()

	def get_name(self):
		""" Return the name of the submodel.
		"""
		return self.name

	def __repr__(self):
		""" Return a description of the model.
		"""
		return 'ClassificationSubmodel({})'.format(str(self.num_classes))

	def size(self):
		""" Returns the size of the submodel (number of classes).
		"""
		return self.num_classes

	def create(self, **kwargs):
		""" Create the actual (keras.models.Model) submodel.
		"""
		return default_classification_model(num_classes=self.size(), **kwargs)

	def loss(self):
		""" Define a loss function for the regression submodel.
		"""
		return focal()
