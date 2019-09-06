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

import tensorflow as tf


class RedirectModel(tf.keras.callbacks.Callback):
	"""Callback which wraps another callback, but executed on a different model.
	```python
	model = keras.models.load_model('model.h5')
	model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
	parallel_model = multi_gpu_model(model, gpus=2)
	parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
	```
	Args
		callback : callback to wrap.
		model    : model to use when executing callbacks.
	"""

	def __init__(
		self,
		callback,
		model
	):
		super(RedirectModel, self).__init__()

		self.callback = callback
		self.redirect_model = model

	def on_epoch_begin(self, epoch, logs=None):
		self.callback.on_epoch_begin(epoch, logs=logs)

	def on_epoch_end(self, epoch, logs=None):
		self.callback.on_epoch_end(epoch, logs=logs)

	def on_batch_begin(self, batch, logs=None):
		self.callback.on_batch_begin(batch, logs=logs)

	def on_batch_end(self, batch, logs=None):
		self.callback.on_batch_end(batch, logs=logs)

	def on_train_begin(self, logs=None):
		# Overwrite the model with our custom model.
		self.callback.set_model(self.redirect_model)

		self.callback.on_train_begin(logs=logs)

	def on_train_end(self, logs=None):
		self.callback.on_train_end(logs=logs)
