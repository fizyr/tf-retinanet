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

from .generator import Generator  # noqa: F401

from ..utils import import_package
from ..utils.anchors import (
	anchor_targets_bbox,
	guess_shapes
)

from ..utils.image import TransformParameters


def preprocess_config(config):
	# If the transform_generator flag is not present, set it to default value.
	if 'transform_generator' not in config:
		from ..utils.transform import random_transform_generator
		config['transform_generator'] = random_transform_generator(flip_x_chance=0.5)

	# If the transform_generator flag is set to none, set it to None.
	if config['transform_generator']  == 'none':
		config['transform_generator'] = None

	# If the transform_generator flag is set to random, set it to the random preset.
	if config['transform_generator']  == 'random':
		from ..utils.transform import random_transform_generator
		config['transform_generator'] = random_transform_generator(
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

	# If the visual_effect_generator flag is not present, set it to default value.
	if 'visual_effect_generator' not in config:
		config['visual_effect_generator'] = None

	# If the visual_effect_generator flag is set to none, set it to None.
	if config['visual_effect_generator']  == 'none':
		config['visual_effect_generator'] = None

	# If the visual_effect_generator flag is set to random, set it to the random preset.
	if config['visual_effect_generator']  == 'random':
		from ..utils.image import random_visual_effect_generator
		config['visual_effect_generator'] = random_visual_effect_generator(
			contrast_range=(0.9, 1.1),
			brightness_range=(-.1, .1),
			hue_range=(-0.05, 0.05),
			saturation_range=(0.95, 1.05)
		)

	# If the batch_size flag is not present, set it to default value.
	if 'batch_size' not in config:
		config['batch_size'] = 1

	# If the group_method flag is not present, set it to default value.
	if 'group_method' not in config:
		config['group_method'] = 'ratio'  # one of 'none', 'random', 'ratio'

	# If the shuffle_groups flag is not present, set it to default value.
	if 'shuffle_groups' not in config:
		config['shuffle_groups'] = True

	# If the image_min_side flag is not present, set it to default value.
	if 'image_min_side' not in config:
		config['image_min_side'] = 800

	# If the image_max_side flag is not present, set it to default value.
	if 'image_max_side' not in config:
		config['image_max_side'] = 1333

	# If the transform_parameters flag is not present, set it to default value.
	if 'transform_parameters' not in config:
		config['transform_parameters'] = TransformParameters()

	# If the transform_parameters flag is set to none, set it to None.
	if config['transform_parameters']  == 'none':
		config['transform_parameters'] = None

	# If the compute_anchor_targets flag is not present, set it to default value.
	if 'compute_anchor_targets' not in config:
		config['compute_anchor_targets'] = anchor_targets_bbox

	# If the compute_shapes flag is not present, set it to default value.
	if 'compute_shapes' not in config:
		config['compute_shapes'] = guess_shapes

	# If the anchors flag is not present, set it to default value.
	if 'anchors' not in config:
		config['anchors'] = None

	return config


def get_generators(config, submodels_manager, preprocess_image, **kwargs):
	generator_pkg = import_package(config['generator']['name'], 'tf_retinanet_generators')

	return generator_pkg.from_config(
		preprocess_config(config['generator']['details']),
		submodels_manager,
		preprocess_image,
		**kwargs
	)
