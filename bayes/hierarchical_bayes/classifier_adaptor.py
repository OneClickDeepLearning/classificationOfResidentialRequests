from sklearn.externals import joblib

class ClassifierAdaptor:
	def __init__(self, model_file):
		self.classifier = None
		self.set_classifier(model_file)


	def set_classifier(self, model_file):
		self.classifier = joblib.load(model_file)

	"""
	@param input is vector after preprocessing
	"""
	def classify(self, input):
		return self.classifier.predict(list([input]))[0]


	def get_predict_proba(self, input):
		return (self.classifier.predict_proba(list([input]))[0]).tolist()


	def get_classes(self):
		return self.classifier.classes_