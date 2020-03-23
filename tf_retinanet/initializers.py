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

from typing import List
import numpy as np
import math

import tensorflow as tf


class PriorProbability(tf.keras.initializers.Initializer):
	""" Apply a prior probability to the weights.
	"""

	def __init__(self, probability: float = 0.01):
		self.probability = probability

	def get_config(self) -> dict:
		return {
			'probability': self.probability
		}

	def __call__(self, shape: List[int], dtype=None) -> np.ndarray:
		# Set bias to -log((1 - p)/p) for foreground.
		if dtype is not None:
			dtype = dtype.as_numpy_dtype()
		result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

		return result
