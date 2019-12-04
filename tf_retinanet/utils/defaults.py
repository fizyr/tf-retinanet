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

import os
from pathlib import Path
from datetime import datetime


default_backbone_config = {
	'details': {
		'weights' : 'imagenet',
		'freeze'  : False,
	}
}


default_callbacks_config = {
	'snapshots_path' : os.path.join(str(Path.home()), 'retinanet-snapshots'),
	'project_name'   : datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
}


default_anchors_config = {
	'sizes'   : [32, 64, 128, 256, 512],
	'strides' : [8, 16, 32, 64, 128],
	'ratios'  : [0.5, 1, 2],
	'scales'  : [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
}


default_generator_config = {
	'details': {
		'anchors'                 : default_anchors_config,
		'batch_size'              : 1,
		'group_method'            : 'ratio',  # None, 'random', 'ratio'
		'image_min_side'          : 800,
		'image_max_side'          : 1333,
		'shuffle_groups'          : True,
		'transform_generator'     : None,     # None, 'basic', 'random'
		'transform_parameters'    : None,     # None or 'standard'
		'visual_effect_generator' : None,     # None or 'random'
	}
}


default_submodels_config = {
	'retinanet': [
		{'category' : 'default_regression',     'details' : {}},
		{'category' : 'default_classification', 'details' : {}},
	]
}


default_train_config = {
	'epochs'              : 50,
	'gpu'                 : 0,
	'lr'                  : 1e-5,
	'max_queue_size'      : 10,
	'steps_per_epoch'     : 10000,
	'use_multiprocessing' : False,
	'weights'             : 'imagenet',
	'workers'             : 1,
}


default_eval_config = {
	'convert_model'   : False,
	'gpu'             : 0,
	'iou_threshold'   : 0.5,
	'max_detections'  : 100,
	'score_threshold' : 0.05,
	'weights'         : None,
}


default_convert_config = {
	'class_specific_filter' : True,
	'nms'                   : True,
}


default_debug_config = {
	'resize'       : True,
	'anchors'      : False,
	'display_name' : False,
	'annotations'  : False,
}


default_training_config = {
	'backbone'  : default_backbone_config,
	'callbacks' : default_callbacks_config,
	'generator' : default_generator_config,
	'submodels' : default_submodels_config,
	'train'     : default_train_config,
}


default_evaluation_config = {
	'backbone'  : default_backbone_config,
	'generator' : default_generator_config,
	'submodels' : default_submodels_config,
	'evaluate'  : default_eval_config,
}


default_conversion_config = {
	'backbone'  : default_backbone_config,
	'generator' : default_generator_config,
	'submodels' : default_submodels_config,
	'convert'   : default_convert_config,
}


default_debugging_config = {
	'backbone'  : default_backbone_config,
	'generator' : default_generator_config,
	'submodels' : default_submodels_config,
	'debug'     : default_debug_config,
}
