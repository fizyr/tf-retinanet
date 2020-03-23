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

from typing import Callable
from typing import Generator as GeneratorType

import numpy as np

from ..models.submodels.manager import SubmodelsManager
from ..utils import import_package
from ..utils.image import VisualEffect
from .generator import Generator  # noqa: F401


def get_transform_generator(transform_generator_type: str) -> GeneratorType[np.ndarray, None, None]:
	""" Constructs a transform generator based on transform generator type.
	Args
		transform_generator_type: Type of the transform generator.
			'basic':  a transform generator which only flips image about x axis.
			'random': a transform generator which applies more than one transformation.

	Returns
		The transform generator.
	"""
	# Set the tranform generator class. If the transform_generator flag is set to basic, use only flip_x.
	if transform_generator_type  == 'basic':
		from ..utils.transform import random_transform_generator
		return random_transform_generator(flip_x_chance=0.5)
	elif transform_generator_type  == 'random':
		from ..utils.transform import random_transform_generator
		return random_transform_generator(
			min_rotation=-0.1,
			max_rotation=0.1,
			min_translation=(-0.1, -0.1),
			max_translation=(0.1, 0.1),
			min_shear=-0.1,
			max_shear=0.1,
			min_scaling=(0.9, 0.9),
			max_scaling=(1.1, 1.1),
			flip_x_chance=0.5,
			flip_y_chance=0.5,
		)
	else:
		return None


def get_visual_effect_generator(visual_effect_generator_type: str) -> GeneratorType[VisualEffect, None, None]:
	""" Constructs a visual effect generator based on visual effect generator type.
	Args
		visual_effect_generator_type: Type of the visual effect generator.
			'random': a visual effect generator which applies more than one visual effects.

	Returns
		The visual effect generator.
	"""

	# If the visual_effect_generator flag is set to random, set it to the random preset.
	if visual_effect_generator_type == 'random':
		from ..utils.image import random_visual_effect_generator
		return random_visual_effect_generator(
			contrast_range=(0.9, 1.1),
			brightness_range=(-.1, .1),
			hue_range=(-0.05, 0.05),
			saturation_range=(0.95, 1.05)
		)
	else:
		return None


def get_generators(
	name: str,
	details: dict,
	submodels_manager: SubmodelsManager,
	preprocess_image: Callable[[np.ndarray], np.ndarray],
	**kwargs
):
	""" Imports generators from an external package,
		and with the retrieved information the submodels manager creates the sumbodels.
		The link between submodels and generators depends on the used generator, hence they are created together in the external package.
	Args
		name:              Module name where the generator is located.
		details:           Details of the generator.
		submodels_manager: Manager containing details for the creation of the submodels.
		preprocess_image:  Function used to preprocess images in the generator.
	Returns
		The specified generators and submodels.
	"""
	generator_pkg = import_package(name, 'tf_retinanet_generators')

	return generator_pkg.from_config(
		details,
		get_transform_generator(details['transform_generator']),
		get_visual_effect_generator(details['visual_effect_generator']),
		submodels_manager,
		preprocess_image,
		**kwargs
	)
