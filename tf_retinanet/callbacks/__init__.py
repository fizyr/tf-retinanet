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

import os


def get_callbacks(
    config,
    model,
    training_model,
    prediction_model,
    validation_generator=None,
    evaluation_callback=None,
    earlystopping=None,
    tensorboard=None,
    reduceLR=None,
):
    """ Returns the callbacks indicated in the config.
	Args
		config              : Dictionary with indications about the callbacks.
		model               : The used model.
		prediction_model    : The used prediction model.
		training_model      : The used training model.
		validation_generator: Generator used during validation.
		evaluation_callback : Callback used to perform evaluation.
		earlystopping		: EarlyStopping criterion callback.
		tensorboard			: Monitor training with TensorBoard.
		reduceLR			: Reduce Learning Rate on plateau callback.
	Returns
		The indicated callbacks.
	"""
    callbacks = []

    # Save snapshots of the model.
    os.makedirs(os.path.join(config["snapshots_path"], config["project_name"]))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            config["snapshots_path"], config["project_name"], "{epoch:02d}.h5"
        ),
        verbose=1,
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    # Evaluate the model.
    if validation_generator:
        if not evaluation_callback:
            raise NotImplementedError("Standard evaluation_callback not implement yet.")
        evaluation_callback = evaluation_callback(validation_generator)
        evaluation_callback = RedirectModel(evaluation_callback, prediction_model)
        callbacks.append(evaluation_callback)

    # Create TensorBoard Callback.
    if config['tensorboard']:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            config["tensorboard_path"]
        )
        callbacks.append(tensorboard_callback)

    # Create Reduce Learning Rate on Plateau Callback.
    if config['reduceLR']:
        reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
            patience=config["reduceLR_patience"]
        )
        callbacks.append(reducer_callback)

    # Create earlystopping callback.
    if config['earlystopping']:
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=config["earlystopping_patience"]
        )
        callbacks.append(earlystopping_callback)

    return callbacks
