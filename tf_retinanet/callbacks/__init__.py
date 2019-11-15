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

from .common import *  # noqa: F401,F403


def get_callbacks(
	config,
	model,
	training_model,
	prediction_model,
	validation_generator=None,
	evaluation_callback=None
):
	callbacks = []

	# Save snapshots of the model.
	os.makedirs(os.path.join(config['snapshots_path'], config['project_name']))
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		os.path.join(
			config['snapshots_path'],
			config['project_name'],
			'{epoch:02d}.h5'
		),
		verbose=1,
	)
	checkpoint = RedirectModel(checkpoint, model)
	callbacks.append(checkpoint)

	# Evaluate the model.
	if validation_generator:
		if not evaluation_callback:
			raise('Standard evaluation_callback not implement yet.')
		evaluation_callback = evaluation_callback(validation_generator)
		evaluation_callback = RedirectModel(evaluation_callback, prediction_model)
		callbacks.append(evaluation_callback)

	return callbacks
