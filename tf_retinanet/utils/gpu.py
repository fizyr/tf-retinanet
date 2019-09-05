import tensorflow as tf

from .version import tf_version_ok


def setup_gpu(gpu_id):
	if tf_version_ok((2, 0, 0)):
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			# Restrict TensorFlow to only use the first GPU.
			try:
				# Currently, memory growth needs to be the same across GPUs.
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)

				# Use only the selcted gpu.
				tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
			except RuntimeError as e:
				# Visible devices must be set before GPUs have been initialized.
				print(e)

			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	else:
		import os
		os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		tf.keras.backend.set_session(tf.Session(config=config))
