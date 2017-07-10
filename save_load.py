import pickle
import os
import copy

def save_model(model, folder_path):
	folder_path = os.path.dirname(folder_path)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	model.train.to_csv(os.path.join(folder_path, "train.csv"))
	model.test.to_csv(os.path.join(folder_path, "test.csv"))

	model_save = copy.copy(model)
	model_save.train = None
	model_save.test = None
	with open(os.path.join(folder_path, "model.pickle"),'wb') as file:
		pickle.dump(model_save, file)

def load_model(folder_path):
	folder_path = os.path.dirname(folder_path)
	if not os.path.exists(folder_path):
		raise Exception("Folder to load model from does not exist.")

	filenames = glob.glob(folder_path + "/*")

	train_path = os.path.join(folder_path, "train.csv")
	test_path = os.path.join(folder_path, "test.csv")
	model_path = os.path.join(folder_path, "model.pickle")

	if not os.path.isfile(train_path) or not os.path.isfile(test_path) or not os.path.isfile(model_path):
		raise Exception("Corrupted model folder.")

	train = pandas.from_csv(train_path)
	test = pandas.from_csv(test_path)
	with open(model_path, "rb") as file:
		model = pickle.load(file)

	model.train = train
	model.test = test

	return model
