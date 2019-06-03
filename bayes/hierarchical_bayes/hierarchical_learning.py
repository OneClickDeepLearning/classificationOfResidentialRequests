from classifier_adaptor import ClassifierAdaptor 
import time


class HierarchicalLearning:
	def __init__(self):
		self.root_classifier = None
		self.sub_classifiers = None
		self.edges_config = None
		self.has_init = False
		self.pred_list = list()
		self.result_01234 = list()
		self.time = 0
		

	def classify(self, input):
		# init
		if self.has_init == False:
			self.params_check()
			self.has_init = True

		start_time = time.time()
		first_layer_result =  self.first_layer_classify(input)
		second_layer_result = self.second_layer_classify(input,first_layer_result)
		end_time = time.time()
		self.time += (end_time - start_time)

		self.pred_list.append(list([first_layer_result, second_layer_result]))
		self.result_01234.append(first_layer_result)
		
		
		return second_layer_result


	def first_layer_classify(self, input):
		return self.root_classifier.classify(input[0])  # return class


	def second_layer_classify(self, input, prev_layer_clf_result):
		# get index of classfier
		index_clf = -1
		for cfg in self.edges_config:
			if prev_layer_clf_result == cfg[0]:
				index_clf = cfg[1]
				break	

		if index_clf == -1:
			raise Exception("edges config error - can not map edge configuration")

		second_layer_classifier = self.sub_classifiers[index_clf]
		second_layer_clf_result = second_layer_classifier.classify(input[index_clf + 1])

		return second_layer_clf_result

		
	def params_check(self):
		if self.root_classifier == None:
			raise Exception("Never set root_classifier")
		if self.sub_classifiers == None:
			raise Exception("Never set sub_classifiers")
		if self.edges_config == None:
			# set default values
			num_sub_clf = len(self.sub_classifiers)
			default_edges_cfg = list()
			for i in range(num_sub_clf):
				tp = tuple((i , i))
				default_edges_cfg.append(tp)
			self.edges_config = default_edges_cfg

		if len(self.sub_classifiers) != len(self.edges_config):
			raise Exception("edges_config & sub_classifiers are not mapping")



	def set_root_classifier(self, root_classifier):
		if not isinstance(root_classifier, ClassifierAdaptor):
			raise Exception("root_classifier should be a ClassifierAdaptor")
		self.root_classifier = root_classifier


	def set_sub_classifiers(self, sub_classifiers):
		if type(sub_classifiers) is not list:
			raise Exception("sub_classifiers should be a list")

		for sub_clf in sub_classifiers:
			if not isinstance(sub_clf, ClassifierAdaptor):
				raise Exception("elements of sub_classifiers should be a ClassifierAdaptor")
		self.sub_classifiers = sub_classifiers


	def set_edges_config(self, edges_config):
		if type(edges_config) is not list:
			raise Exception("edges_config should be a list")
		for cfg in edges_config:
			if cfg is not tuple:
				raise Exception("elements of edges_config should be a tuple")
