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
	""" bla
	"""
	def __init__(self, config):
		""" grfgr
		"""
		self.classification = None
		self.regression = None
		self.submodels = {}
		for submodel in config['submodels']['retinanet']:
			if submodel['type'] == 'default_regression':
				from .regression import BboxRegressionSubmodel, load_annotations, create_batch
				submodel['class']        = BboxRegressionSubmodel
				submodel['loader']       = load_annotations
				submodel['create_batch'] = create_batch
				self.regression = submodel
				continue
			elif submodel['type'] == 'default_classification':
				from .classification import ClassificationSubmodel, load_annotations, create_batch
				submodel['class']        = ClassificationSubmodel
				submodel['loader']       = load_annotations
				submodel['create_batch'] = create_batch
				self.classification = submodel
				continue
			else:
				try:
					submodel_pkg = __import__('tf_retinanet_submodels', fromlist=[submodel['type']])
					submodel_pkg = getattr(submodel_pkg, submodel['type'])
				except ImportError:
					raise(submodel['type'] + 'is not a valid submodel')
				submodel['class'] = submodel_pkg.from_config()
				if 'main_classification' in submodel and submodel['main_classification']:
					self.classification = submodel
				if 'main_regression' in submodel and submodel['main_regression']:
					self.regression = submodel
				self.submodels[submodel['name']] = submodel #remove name from here

		if not self.classification:
			raise("Could not find main classification submodel.")
		if not self.regression:
			raise("Could not find main regression submodel.")

		if 'details' in self.classification and 'classes' in self.classification['details']:
			self.classes = self.classification['details']['classes']
		else:
			self.classes = None


	def num_classes(self):
		if self.classes:
			return len(self.classes)
		return None


	def create(self, num_classes=None):
		if num_classes and self.num_classes():
			raise("Number of classes provided twice and conflicting.")
		elif not num_classes and not self.num_classes():
			raise("Number of classes not provided.")

		if num_classes is None:
			num_classes = self.num_classes()

		submodels = []
		self.regression['class'] = self.regression['class']()
		submodels.append(self.regression['class'])
		self.classification['class'] = self.classification['class'](num_classes=num_classes)
		submodels.append(self.classification['class'])

		for submodel in self.submodels:
			submodels.append(submodel['class']())

		return submodels

