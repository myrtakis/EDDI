import numpy as np
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from torch import nn


def class_swap(
	y: np.ndarray,
	seed: int,
):
	"""
	This function directly affects the posterior p(y|X) inducing severe drift.
	The classes are completely swapped in a structured way.
	"""
	error_col = np.zeros(len(y), dtype=int)
	y_drifted = y.copy()
	drift_start_index = len(y) // 2
	drift_end_index = len(y)
	error_distr = np.ones(drift_end_index - drift_start_index)
	swap_map = {}
	unq_labels = np.unique(y[drift_start_index:drift_end_index])
	for i, c in enumerate(unq_labels):
		np.random.seed(i + seed)
		class_to_swap = np.random.choice(np.setdiff1d(unq_labels, c), size=1)[0]
		swap_map[int(c)] = int(class_to_swap)
	for i, ind in enumerate(range(drift_start_index, drift_end_index)):
		label_changed = int(np.random.binomial(1, p=error_distr[i]))
		error_col[ind] = label_changed
		if label_changed:
			y_drifted[ind] = swap_map[y[ind]]
	return y_drifted, error_col


def induce_imbalance(
	drift_speed: str,
	drift_start: str,
	X: np.ndarray,
	y: np.ndarray,
	seed: int,
	train_size=0.8,
	majority_class_downsample_ratio=0,
):
	"""
	The drift is induced by downsampling the majority class c in the training phase
	and oversampling c in the evaluation phase. The goal is to find the drift in the evaluation phase.

	Parameters
	----------
	drift_speed
	seed

	Returns
	-------
	"""

	label_counts = np.unique(y, return_counts=True)
	max_count_id = np.argmax(label_counts[1])
	majority_class = label_counts[0][max_count_id]

	maj_class_ids = np.where(y == majority_class)[0]
	other_train_class_ids = np.where(y != majority_class)[0]

	dh_d = read_split_dataset(
		X=X[other_train_class_ids],
		y=y[other_train_class_ids],
		train_size=train_size,
		val_size=0.05,
		seed=seed,
	)

	if dh_d is None:
		return None, None

	X_maj = X[maj_class_ids]
	y_maj = y[maj_class_ids]
	num_of_maj_ids_to_keep = int(
		len(maj_class_ids) * majority_class_downsample_ratio
	)

	dh_d.X_train = np.vstack((dh_d.X_train, X_maj[:num_of_maj_ids_to_keep]))
	dh_d.y_train = np.concatenate((dh_d.y_train, y_maj[:num_of_maj_ids_to_keep]))

	maj_class_X_test = X_maj[num_of_maj_ids_to_keep:]
	maj_class_y_test = y_maj[num_of_maj_ids_to_keep:]

	test_new_total = len(dh_d.y_test) + len(maj_class_X_test)

	drift_start_index = 0 if drift_start == "early" else min(test_new_total // 2, test_new_total - len(maj_class_X_test))
	drift_end_index = (
		test_new_total // 2 if drift_start == "early" else test_new_total
	)
	if drift_speed == "gradual":
		error_distr = np.linspace(0, 1, drift_end_index - drift_start_index)
	else:
		error_distr = np.ones(drift_end_index - drift_start_index)

	X_test_d, y_test_d = [], []
	maj_counter = 0
	clean_test_counter = 0

	for i in range(test_new_total):
		if i >= drift_start_index:
			if i >= drift_end_index:
				add_maj_sample = 1
			else:
				add_maj_sample = int(np.random.binomial(1, p=error_distr[i-drift_start_index]))
			if add_maj_sample and maj_counter < len(maj_class_X_test):
				X_test_d.append(maj_class_X_test[maj_counter])
				y_test_d.append(maj_class_y_test[maj_counter])
				maj_counter += 1
			else:
				if clean_test_counter < len(dh_d.X_test):
					X_test_d.append(dh_d.X_test[clean_test_counter])
					y_test_d.append(dh_d.y_test[clean_test_counter])
					clean_test_counter += 1
		else:
			if clean_test_counter < len(dh_d.X_test):
				X_test_d.append(dh_d.X_test[clean_test_counter])
				y_test_d.append(dh_d.y_test[clean_test_counter])
				clean_test_counter += 1

	if maj_counter < len(maj_class_X_test):
		X_test_d.extend(maj_class_X_test[maj_counter:].tolist())
		y_test_d.extend(maj_class_y_test[maj_counter:].tolist())

	if clean_test_counter < len(dh_d.X_test):
		X_test_d.extend(dh_d.X_test[clean_test_counter:].tolist())
		y_test_d.extend(dh_d.y_test[clean_test_counter:].tolist())

	dh_d.X_test = np.array(X_test_d)
	dh_d.y_test = np.array(y_test_d)
	error_col = np.zeros(len(dh_d.y_test), dtype=int)
	error_col[np.where(dh_d.y_test == majority_class)[0]] = 1
	return dh_d, error_col


	def add_adversarial_noise(
			X_test: np.ndarray,
			y_test: np.ndarray,
			model: nn.Module,
			num_classes: int,
			seed: int,
			device='cpu'
	):
		loss = nn.CrossEntropyLoss()
		model.eval()

		X = X_test.copy().astype(dtype=np.float32)
		y = y_test.copy()

		classifier = PyTorchClassifier(
			model=model.to(device),
			clip_values=(np.min(X), np.max(X)),
			loss=loss,
			optimizer=None,
			input_shape=X.shape[1:],
			nb_classes=num_classes,
			device_type='gpu' if 'cuda' in device else 'cpu',
		)

		attack = CarliniL2Method(
			classifier=classifier, verbose=True, confidence=0.1, batch_size=64, max_iter=5
		)

		predictions_adv = classifier.predict(X)
		pred_labels = np.argmax(predictions_adv, axis=1)
		accuracy = np.sum(pred_labels == y_test) / len(y_test)
		print("Accuracy on benign test examples: {}%".format(int(accuracy * 100)))

		np.random.seed(seed)

		# Step 6: Generate adversarial test examples
		X_test_adv = attack.generate(x=X)
		predictions_adv = classifier.predict(X_test_adv)
		pred_adv_labels = np.argmax(predictions_adv, axis=1)
		accuracy = np.sum(pred_adv_labels == y_test) / len(pred_adv_labels)
		print("Accuracy on adversarial test examples: {}%".format(int(accuracy * 100)))

		adv_ids = np.where(pred_adv_labels != y_test)[0]
		X_test_adv = X_test_adv[adv_ids]
		y_test_adv = pred_adv_labels[adv_ids]
		return X_test_adv, y_test_adv, adv_ids
