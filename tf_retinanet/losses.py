import tensorflow as tf


def focal(alpha=0.25, gamma=2.0):
	""" Create a functor for computing the focal loss.

	Args
		alpha: Scale the focal weight with alpha.
		gamma: Take the power of the focal weight with gamma.

	Returns
		A functor that computes the focal loss using the alpha and gamma.
	"""
	def _focal(y_true, y_pred, **kwargs):
		""" Compute the focal loss given the target tensor and the predicted tensor.
		As defined in https://arxiv.org/abs/1708.02002
		Args
			y_true: Tensor of target data from the generator with shape (B, N, num_classes).
			y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
		Returns
			The focal loss of y_pred w.r.t. y_true.
		"""
		labels         = y_true[:, :, :-1]
		anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
		classification = y_pred

		# filter out "ignore" anchors
		indices        = tf.where(tf.keras.backend.not_equal(anchor_state, -1))
		labels         = tf.gather_nd(labels, indices)
		classification = tf.gather_nd(classification, indices)

		# compute the focal loss
		alpha_factor = tf.keras.backend.ones_like(labels) * alpha
		alpha_factor = tf.where(tf.keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
		focal_weight = tf.where(tf.keras.backend.equal(labels, 1), 1 - classification, classification)
		focal_weight = alpha_factor * focal_weight ** gamma

		cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels, classification)

		# compute the normalizer: the number of positive anchors
		normalizer = tf.where(tf.keras.backend.equal(anchor_state, 1))
		normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
		normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

		return tf.keras.backend.sum(cls_loss) / normalizer

	return _focal


def smooth_l1(sigma=3.0):
	""" Create a smooth L1 loss functor.

	Args
		sigma: This argument defines the point where the loss changes from L2 to L1.

	Returns
		A functor for computing the smooth L1 loss given target data and predicted data.
	"""
	sigma_squared = sigma ** 2

	def _smooth_l1(y_true, y_pred, **kwargs):
		""" Compute the smooth L1 loss of y_pred w.r.t. y_true.

		Args
			y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
			y_pred: Tensor from the network of shape (B, N, 4).

		Returns
			The smooth L1 loss of y_pred w.r.t. y_true.
		"""
		# Separate target and state.
		regression        = y_pred
		regression_target = y_true[:, :, :4]
		anchor_state      = y_true[:, :, 4]

		# Filter out "ignore" anchors.
		indices           = tf.where(tf.keras.backend.equal(anchor_state, 1))
		regression        = tf.gather_nd(regression, indices)
		regression_target = tf.gather_nd(regression_target, indices)

		# Compute smooth L1 loss
		# f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
		#        |x| - 0.5 / sigma / sigma    otherwise
		regression_diff = regression - regression_target
		regression_diff = tf.keras.backend.abs(regression_diff)
		regression_loss = tf.where(
			tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
			0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
			regression_diff - 0.5 / sigma_squared
		)

		# compute the normalizer: the number of positive anchors
		normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
		normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
		return tf.keras.backend.sum(regression_loss) / normalizer

	return _smooth_l1
