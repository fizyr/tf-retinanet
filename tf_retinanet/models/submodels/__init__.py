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

from typing import List, Callable, Tuple

import tensorflow as tf
import numpy as np

from .manager import SubmodelsManager
from ...utils.image import TransformParameters

class Submodel():
	""" Abstract class for all submodels.
	"""

	def get_name(self) -> str:
		""" The name of the submodel.
		"""
		raise NotImplementedError()

	def size(self) -> int:
		""" The size of the submodel.
		This generally means the number of values per anchor / roi.
		"""
		raise NotImplementedError()

	def create(self, **kwargs):
		""" Create a tf.keras.Model out of the submodel information.
		"""
		raise NotImplementedError()

	def check(self, annotation: dict) -> bool:
		""" Check if the annotation makes sense for this submodel.
		The default version simply return true.
		"""
		return True

	def loss(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
		""" Return the loss functions to use for this submodel.
		"""
		raise NotImplementedError()

	def get_evaluation(self):
		""" return the custom evaluation procedure.
		"""
		return None

	def get_evaluation_callback(self) -> tf.keras.callbacks.Callback:
		""" return the custom evaluation callback.
		"""
		return None

	def random_transform(self, image: np.ndarray, annotations: List[dict], transform: np.ndarray, transform_parameters: TransformParameters) -> Tuple[np.ndarray, dict]:
		""" Transform the annotations based on a transformation matrix.
		"""
		raise NotImplementedError()

	def preprocess(self, image: np.ndarray, annotations: List[dict], image_min_side: int, image_max_side: int) -> Tuple[np.ndarray, dict]:
		""" Preprocesses image and annotations.
		"""
		raise NotImplementedError()

	def load_annotations(self, annotations: dict, meta: dict, image_meta: dict) -> dict:
		""" Load annotations from meta.
		"""
		return annotations

	def create_batch(self, anchors: np.ndarray, annotations: dict, positive_indices: np.ndarray, ignore_indices: np.ndarray, overlap_indices: np.ndarray) -> np.ndarray:
		""" Create a batch to train on.
		"""
		raise NotImplementedError()

	@staticmethod
	def get_custom_objects() -> dict:
		""" Get the custom objects needed by the submodel.
		"""
		return {}
