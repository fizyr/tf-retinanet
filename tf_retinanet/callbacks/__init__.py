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
from typing import List

import tensorflow as tf

from .common import *  # noqa: F401,F403
from ..generators.generator import Generator


def get_callbacks(
	snapshots_path: str,
	project_name: str,
	model: tf.keras.Model,
	training_model: tf.keras.Model,
	prediction_model: tf.keras.Model,
	validation_generator: Generator = None,
	evaluation_callback: Generator = None,
) -> List[tf.keras.callbacks.Callback]:
	""" Returns the callbacks used for training.

	Args
		snapshots_path      : The path to save snapshots to.
		project_name        : The name of the project, which will be used to create a directory.
		model               : The used model.
		training_model      : The used training model.
		prediction_model    : The used prediction model.
		validation_generator: Generator used during validation.
		evaluation_callback : Callback used to perform evaluation.

	Returns
		The indicated callbacks.
	"""
	callbacks = []

	# Save snapshots of the model.
	os.makedirs(os.path.join(snapshots_path, project_name))
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		os.path.join(
			snapshots_path,
			project_name,
			'{epoch:02d}.h5'
		),
		verbose=1,
	)
	checkpoint = RedirectModel(checkpoint, model)
	callbacks.append(checkpoint)

	# Evaluate the model.
	if validation_generator:
		if not evaluation_callback:
			raise NotImplementedError('Standard evaluation_callback not implement yet.')
		evaluation_callback = evaluation_callback(validation_generator)
		evaluation_callback = RedirectModel(evaluation_callback, prediction_model)
		callbacks.append(evaluation_callback)

	return callbacks
