"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
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

from .manager import SubmodelsManager

class Submodel(object):
	""" Abstract class for all submodels.
	"""
	def __init__(self, annotation_source=None):
		""" Constructor for submodel.
		Args
			annotation_source: Defines where in the annotations to get the data from.
		"""
		self.annotation_source = annotation_source
		super(Submodel, self).__init__()


	def size(self):
		""" The size of the submodel.
		This generally means the number of values per anchor / roi.
		"""
		raise NotImplementedError()


	def create(self, **kwargs):
		""" Create a keras.Model out of the submodel information.
		"""
		raise NotImplementedError()


	def check(self, annotation):
		""" Check if the annotation makes sense for this submodel.
		The default version simply return true.
		"""
		return True


	def loss(self):
		""" Return the loss functions to use for this submodel.
		"""
		raise NotImplementedError()


	def random_transform(self, image, annotations, transform, transform_parameters):
		""" Transform the annotations based on a transformation matrix.
		"""
		raise NotImplementedError()


	def preprocess(self, image, annotations, image_min_side, image_max_side):
		""" Preprocesses image and annotations.
		"""
		raise NotImplementedError()


	def load_annotations(self, annotations, meta, image_meta):
		""" Load annotations from meta.
		"""
		return annotations


	def create_batch(self, anchors, annotations, positive_indices, ignore_indices, overlap_indices):
		""" Create a batch to train on.
		"""
		raise NotImplementedError()

	@staticmethod
	def get_custom_objects():
		""" Get the custom objects needed by the submodel.
		"""
		return {}


def preprocess_config(config):
	return config


def get_submodels(config, **kwargs):
	submodels = []
	submodels_names = [submodel['type'] for submodel in config['submodels']]
	if 'default_regression' in submodels_names:
		submodels_names.remove('default_regression')
		from .regression import BboxRegressionSubmodel
		submodels.append(BboxRegressionSubmodel)
	if 'default_classification' in submodels_names:
		submodels_names.remove('default_classification')
		from .classification import ClassificationSubmodel
		submodels.append(ClassificationSubmodel)
	for submodel_name in submodels_names:
		try:
			submodel_pkg = __import__('tf_retinanet_submodels', fromlist=[submodel_name])
			submodel_pkg = getattr(submodel_pkg, submodel_name)
		except ImportError:
			raise(submodel_name + 'is not a valid submodel')
		submodels.append(submodel_pkg.from_config(
			preprocess_config(config['submodels']['details']),
			**kwargs
		))
	return submodels
