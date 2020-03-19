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

from typing import Dict, List, Tuple, Callable

from ...utils import import_package
from ...utils.defaults import default_submodels_config

import tensorflow as tf


class SubmodelsManager():
	""" Class that parses submodels from configuration and creates them.
	"""
	def __init__(self, submodel_configs: List[Dict[str, dict]] = None):
		""" Initialize the manager.
		Args:
			config : configuration dictionary.
					 It should contain a list of submodels, each of which should be a dictionary containing:
						category : The category of submodel to be parsed (bbox_regression, classification, etc.).
						details  : A dictionary with details about the submodel. Refer to each submodel class for more info.
					 If not specified, default classification and bbox regression will be used.
		"""
		if submodel_configs is None:
			submodel_configs = default_submodels_config

		self.classification       = None
		self.regression           = None
		self.additional_submodels = []

		# Loop through the specified submodels.
		for submodel in submodel_configs:
			# Add details key, if not specified. Each submodel will fill it with its defaults.
			if 'details' not in submodel:
				submodel['details'] = {}

			# Parse the submodels.
			if 'category' not in submodel:
				raise ValueError("A submodel category was not specified.")
			elif submodel['category'] == 'default_regression':
				from .regression import BboxRegressionSubmodel
				self.regression = BboxRegressionSubmodel(**submodel['details'])
				continue
			elif submodel['category'] == 'default_classification':
				from .classification import ClassificationSubmodel
				self.classification = ClassificationSubmodel(**submodel['details'])
				continue
			else:
				# Parse submodels from external package.
				submodel_package  = import_package(submodel['category'], 'tf_retinanet_submodels')
				submodel['class'] = submodel_package.parse_submodel(submodel['details'])
				# If the submodel is indicated as main, set it as such in the local submodels.
				if 'main_classification' in submodel and submodel['main_classification']:
					self.classification = submodel
					continue
				if 'main_regression' in submodel and submodel['main_regression']:
					self.regression = submodel
					continue
				self.additional_submodels.append(submodel)

		# We need at least a main classification and a regression submodel to build RetinaNet.
		if not self.classification:
			raise ValueError("Could not find main classification submodel.")
		if not self.regression:
			raise ValueError("Could not find main regression submodel.")

	def create(self) -> List[Tuple[str, tf.keras.Model]]:
		""" Initialize the submodels classes that were provided.
		"""
		# Initialize and append all provided submodels.
		submodels = []
		submodels.append((self.regression.get_name(), self.regression.create()))
		submodels.append((self.classification.get_name(), self.classification.create()))
		for submodel in self.additional_submodels:
			submodels.append((submodel.get_name(), submodel.create()))

		return submodels

	def losses(self) -> Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
		""" Initialize the submodels classes that were provided.
		"""
		# Initialize and append all provided submodels.
		losses = {}
		losses[self.regression.get_name()] = self.regression.loss()
		losses[self.classification.get_name()] = self.classification.loss()
		for submodel in self.additional_submodels:
			losses[submodel.get_name()] = submodel.loss()

		return losses

	#def get_evaluation(self):
	#	""" Get evaluation procedure from submodels, or use default.
	#	"""
	#	evaluations = []
	#	for submodel in self.submodels:
	#		evaluations.append(submodel.get_evaluation()) if submodel.get_evaluation() is not None else []

	#	assert (len(evaluations) < 2), "More than one evaluation procedure has been provided."

	#	if evaluations:
	#		return evaluations[0]
	#	else:
	#		from ...utils.eval import evaluate
	#		return evaluate

	#def get_evaluation_callback(self):
	#	""" Get evaluation callback from submodels, or use default.
	#	"""
	#	callbacks = []
	#	for submodel in self.submodels:
	#		callbacks.append(submodel.get_evaluation_callback()) if submodel.get_evaluation_callback() is not None else []

	#	assert (len(callbacks) < 2), "More than one evaluation callback has been provided."

	#	if callbacks:
	#		return callbacks[0]
	#	else:
	#		from ...callbacks.eval import Evaluate
	#		return Evaluate
