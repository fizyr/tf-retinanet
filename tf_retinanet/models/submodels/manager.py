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

class SubmodelsManager(object):
	""" Class that parses submodels from configuration and creates them.
	"""
	def __init__(self, config):
		""" Initialize the manager.
		Args:
			config: configuration dictionary.
		"""
		self.classification       = None
		self.regression           = None
		self.additional_submodels = []


		# Loop through the specified submodels.
		for submodel in config['submodels']['retinanet']:
			if 'details' not in submodel:
				submodel['details'] = {}

			# Parse the default submodels.
			if submodel['type'] == 'default_regression':
				from .regression import BboxRegressionSubmodel
				submodel['class'] = BboxRegressionSubmodel
				self.regression = submodel
				continue
			elif submodel['type'] == 'default_classification':
				from .classification import ClassificationSubmodel
				submodel['class'] = ClassificationSubmodel
				self.classification = submodel
				continue
			else:
				# Search the indicated submodels in external packages.
				try:
					submodel_pkg = __import__('tf_retinanet_submodels', fromlist=[submodel['type']])
					submodel_pkg = getattr(submodel_pkg, submodel['type'])
				except ImportError:
					raise(submodel['type'] + 'is not a valid submodel')
				submodel['class'] = submodel_pkg.from_config(submodel['details'])
				# If the submodel is indicated as main, set it in the local submodels.
				if 'main_classification' in submodel and submodel['main_classification']:
					self.classification = submodel
					continue
				if 'main_regression' in submodel and submodel['main_regression']:
					self.regression = submodel
					continue
				self.additional_submodels.append(submodel)

		# We need at least a main classification and a regression submodel to build RetinaNet.
		if not self.classification:
			raise("Could not find main classification submodel.")
		if not self.regression:
			raise("Could not find main regression submodel.")

		# Parse the classes, if provided.
		if 'classes' in self.classification['details']:
			self.classes = self.classification['details']['classes']
		else:
			self.classes = None

	def num_classes(self):
		""" If classes are provided in the configuration file, return number of classes.
		"""
		if self.classes:
			return len(self.classes)
		else:
			return None

	def create(self, num_classes=None):
		""" Create the submodels that were provided.
		Args:
			num_classes: number of classification classes.
		"""
		# If the number of classes is provided, add the information to the classification details.
		if num_classes:
			self.classification['details']['num_classes'] = num_classes

		# Instantiate main regression and classification submodels.
		self.regression     = self.regression['class'](self.regression['details'])
		self.classification = self.classification['class'](self.classification['details'])

		# Instantiate and append all provided submodels.
		self.submodels = []
		self.submodels.append(self.regression)
		self.submodels.append(self.classification)
		for submodel in self.additional_submodels:
			self.submodels.append(submodel['class'](submodel['details']))

	def get_submodels(self):
		""" Return the submodels.
		"""
		return self.submodels
