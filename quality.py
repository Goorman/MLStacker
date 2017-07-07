from sklearn.metrics import log_loss

def measure_logloss(model, dataset, target, prediction, **logloss_params):
	if dataset == "train":
		target = model.train[target]
		prediction = model.train[prediction]
	elif dataset == "test":
		target = model.test[target]
		prediction = model.test[prediction]
	else:
		raise Exception("Dataset must be train or test.")

	return log_loss(target, prediction, **logloss_params)
