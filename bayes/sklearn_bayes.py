import pickle
import numpy as np
import math
import time

from bow import get_bag_of_word_vector
from bow import get_testset_bow_vector

from sklearn.metrics import roc_curve, auc, log_loss
from scipy import interp
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# import matplotlib.pyplot as plt

current = "8w_zkall"

with open("data/" + current + "/trainset_bow_instance_label.pickle", 'rb') as ilf:
	trainset_instance_labels = pickle.load(ilf)

with open("data/" + current + "/trainset_bow_instance_classes.pickle", 'rb') as icf:
	trainset_instance_classes = pickle.load(icf)

with open("data/" + current + "/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list = pickle.load(clf)

with open("data/blind_test/testset_bow_instance_label.pickle", 'rb') as ilf:
	testset_instance_labels = pickle.load(ilf)

with open("data/blind_test_01234/testset_bow_instance_classes.pickle", 'rb') as icf:
	testset_instance_classes = pickle.load(icf)

# length of four classes
length_classes = list()
for class_list in trainset_classes_list:
	length_classes.append(len(class_list))



def get_instances_vector_label(data_set):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list
	global length_classes


	X = list()
	Y = list()
	for iid in data_set:
		# normalize classes val
		classes_vec = list()
		classes_idxes = trainset_instance_classes[iid]
		for idx in range(0, 4):
			class_idx = classes_idxes[idx]
			dimen = [0] * length_classes[idx]
			dimen[class_idx] = 1
			classes_vec += dimen

		bow_vec = get_bag_of_word_vector(iid)

		x = classes_vec + bow_vec
		y = int(trainset_instance_labels[iid])
		X.append(x)
		Y.append(y)

		print("INFO: x-class: " + str(len(classes_vec)))
		print("INFO: bow: " + str(len(bow_vec)))

	return X,Y


def get_test_instance_vector_label(test_set):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list
	global length_classes

	X = list()
	Y = list()
	for iid in test_set:
		# normalize classes val
		classes_vec = list()
		for index in range(0, 4):
			dimen = [0] * length_classes[index]
			all_classes = trainset_classes_list[index]
			class_val = testset_instance_classes[iid][index]
			if class_val in all_classes:
				dimen_idx = all_classes.index(class_val)
				dimen[dimen_idx] = 1

			classes_vec += dimen

		# bow_val
		bow_vec = get_testset_bow_vector(iid)

		x = classes_vec + bow_vec
		y = int(testset_instance_labels[iid])
		X.append(x)
		Y.append(y)

	return X, Y


def validate(Y_pred, Y_true, instance_ids):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list
	global length_classes

	correct_count = 0

	for idx in range(0, len(Y_true)):
		y_true = Y_true[idx]
		y_pred = Y_pred[idx]
		instance_id = instance_ids[idx]

		#print( "[REPORT] instance_id: " + instance_id + " | true_label : " + str(y_true) + " | pred_label : " + str(y_pred) )
		
		if y_true == y_pred:
			correct_count += 1
		
	print("------------ Accuracy : " + str(correct_count / len(Y_true)) + "------------")



def grid_search(train_set):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list
	global length_classes


	X_train, Y_train = get_instances_vector_label(train_set)

	params = {
		'alpha': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
	}

	nb_classifier = BernoulliNB(alpha=1, fit_prior=True)
	grid_search = GridSearchCV(nb_classifier, param_grid=params, scoring='accuracy', cv=5)
	grid_search.fit(X_train, Y_train)
	best_parames = grid_search.best_estimator_.get_params()

	print("-------------- Best score : " + str(grid_search.best_score_) + "--------------")
	print("-------------- Best params --------------")
	for p, v in best_parames.items():
		print(p + " : " + str(v))


def roc_auc(data_set):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list

	X, Y = get_instances_vector_label(data_set)
	Y = label_binarize(Y, classes=[0, 1, 2, 3, 4])
	n_classes = Y.shape[1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

	classifiers = OneVsRestClassifier(BernoulliNB(alpha=1, fit_prior=True))
	Y_score = classifiers.fit(X_train, Y_train).predict_proba(X_test)

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), 
		color='deeppink', linestyle=':', linewidth=4)
	plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), 
		color='navy', linestyle=':', linewidth=4)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.show()



# the func is to train the model to pred the query without 4 pre-classes
def get_instances_vector_label_model(data_set):
	global trainset_instance_labels
	global trainset_instance_classes
	global trainset_classes_list


	X = list()
	Y = list()
	for iid in data_set:
		bow_vec = get_bag_of_word_vector(iid)
		x = bow_vec
		y = int(trainset_instance_labels[iid])
		X.append(x)
		Y.append(y)

	return X,Y
 


if __name__ == '__main__':
	#instance_list = list(trainset_instance_labels.keys())
	test_list = list(testset_instance_labels.keys())

	# testing
	#nb_classifier = BernoulliNB(alpha=0.2, fit_prior=True)
	#X_train, Y_train = get_instances_vector_label(instance_list)
	X_test, Y_test = get_test_instance_vector_label(test_list)

	start_time = time.time()

	#nb_classifier.fit(X_train, Y_train)
	nb_classifier = joblib.load("")
	Y_pred = nb_classifier.predict(X_test)
	validate(Y_pred, Y_test, test_list)

	end_time = time.time()

	joblib.dump(nb_classifier, current + '_classifier.joblib')
	print("------------ Model generated ------------")
	print(str(end_time - start_time) + "s")

	print("================= log loss ===================")
	y_prob = nb_classifier.predict_proba(X_test)

	print(y_prob.shape)
	
	print(str(log_loss(Y_test, y_prob, labels=np.arange(y_prob.shape[1]))))

	# split data set
	# size = math.ceil(len(instance_list) / 5)
	# chunks = list()

	# for c in range(0, 4):
	# 	subset = instance_list[ c * size : c * size + size]
	# 	chunks.append(subset)
	# # last chunk
	# subset = instance_list[4 * size :]
	# chunks.append(subset)


	# cross validation
	# for i in range(0, 5):
	# 	# build test set & training set
	# 	test_set = chunks[i]
	# 	training_set = list()
	# 	for j in range(0, 5):
	# 		if j != i:
	# 			training_set = training_set + chunks[j]

	#  	# training
	# 	print("[INFO] training set size :" + str(len(training_set)))
	# 	print("[INFO] test set size :" + str(len(test_set)))
		
	# 	nb_classifier = BernoulliNB(alpha=0.2, fit_prior=True)
	# 	X_train, Y_train = get_instances_vector_label(training_set)
	# 	X_test, Y_test = get_instances_vector_label(test_set)

	# 	nb_classifier.fit(X_train, Y_train)
	# 	Y_pred = nb_classifier.predict(X_test)
	# 	validate(Y_pred, Y_test, test_set)


	# one run
	# test_set = chunks[0]
	# train_set = chunks[1] + chunks[2] + chunks[3] + chunks[4]

	# # # test_set = test_set[:1000]
	# # # grid_search(test_set)
	
	# nb_classifier = BernoulliNB(alpha=0.2, fit_prior=True)
	# X_train, Y_train = get_instances_vector_label(train_set)
	# X_test, Y_test = get_instances_vector_label(test_set)

	# nb_classifier.fit(X_train, Y_train)
	# Y_pred = nb_classifier.predict(X_test)

	# validate(Y_pred, Y_test)

	# roc auc
	#roc_auc(instance_list)


	# grid search
	# data_set = chunks[0] + chunks[2] + chunks[4]
	# grid_search(data_set)


	# train the model without 4-preclasses
	# nb_classifier = BernoulliNB(alpha=0.2, fit_prior=True)
	# X_train, Y_train = get_instances_vector_label_model(instance_list)
	# nb_classifier.fit(X_train, Y_train)
	# joblib.dump(nb_classifier, 'zk4_classifier.joblib')
	# print("------------ Model generated ------------")
