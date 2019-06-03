import pickle
import numpy as np
import math
import time

import matplotlib.pyplot as plt
from bow import get_testset_bow_vector
from classifier_adaptor import ClassifierAdaptor
from hierarchical_learning import HierarchicalLearning
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score



with open("../data/8w_01234/trainset_bow_instance_classes.pickle", 'rb') as icf:
	trainset_instance_classes = pickle.load(icf)

with open("../data/8w_01234/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_01234 = pickle.load(clf)
with open("../data/8w_zk0/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_zk0 = pickle.load(clf)
with open("../data/8w_zk1/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_zk1 = pickle.load(clf)
with open("../data/8w_zk2/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_zk2 = pickle.load(clf)
with open("../data/8w_zk3/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_zk3 = pickle.load(clf)
with open("../data/8w_zk4/trainset_bow_classes_list.pickle", 'rb') as clf:
	trainset_classes_list_zk4 = pickle.load(clf)


with open("../data/4k_test01234/testset_bow_instance_label.pickle", 'rb') as ilf:
	_testset_instance_labels = pickle.load(ilf)

with open("../data/4k_test/testset_bow_instance_label.pickle", 'rb') as ilf:
	testset_instance_labels = pickle.load(ilf)

with open("../data/4k_test/testset_bow_instance_classes.pickle", 'rb') as icf:
	testset_instance_classes = pickle.load(icf)

# length of four classes
sec_layer_len_classes = list()
sec_layer_classes_list = list()

length_classes = list()
for class_list in trainset_classes_list_01234:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_01234)

length_classes = list()
for class_list in trainset_classes_list_zk0:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_zk0)

length_classes = list()
for class_list in trainset_classes_list_zk1:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_zk1)

length_classes = list()
for class_list in trainset_classes_list_zk2:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_zk2)

length_classes = list()
for class_list in trainset_classes_list_zk3:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_zk3)

length_classes = list()
for class_list in trainset_classes_list_zk4:
	length_classes.append(len(class_list))
sec_layer_len_classes.append(length_classes)
sec_layer_classes_list.append(trainset_classes_list_zk4)



def get_test_instance_vector_label(test_set):
	global real_list
	global Y_01234

	X = list()
	Y = list()
	cc = ['01234', 'zk0', 'zk1', 'zk2', 'zk3', 'zk4']
	for iid in test_set:

		x_list = list()
		# normalize classes val
		for iii in range(0, 6):

			classes_vec = list()
			for index in range(0, 4):
				dimen = [0] * sec_layer_len_classes[iii][index]
				all_classes = sec_layer_classes_list[iii][index]
				class_val = testset_instance_classes[iid][index]
				if class_val in all_classes:
					dimen_idx = all_classes.index(class_val)
					dimen[dimen_idx] = 1

				classes_vec += dimen

			# bow_val
			x = classes_vec + get_testset_bow_vector(iid, cc[iii])
			x_list.append(x)


		y = int(testset_instance_labels[iid])
		_y = int(_testset_instance_labels[iid])
		
		Y_01234.append(_y)
		X.append(x_list)
		Y.append(y)

		real_list.append(list([_y, y]))

	return X, Y


def validate(Y_pred, Y_true, instance_ids):
	correct_count = 0

	for idx in range(0, len(Y_true)):
		y_true = Y_true[idx]
		y_pred = Y_pred[idx]
		instance_id = instance_ids[idx]

		print( "[REPORT] instance_id: " + instance_id + " | true_label : " + str(y_true) + " | pred_label : " + str(y_pred) )
		
		if y_true == y_pred:
			correct_count += 1
	print("------------ Accuracy : " + str(correct_count / len(Y_true)) + "------------")



def hierarchical_acc(pred_list, real_list, n=1):
        # pred_list: list of predicted label path       exp:[[4,121],[1,6],[2,36],[2,33]...]
        # real_list: list of real label path            exp:[[4,124],[1,6],[3,65],[2,33]...]
        # Returns hP, hR, F-n score                     dtype: float
        INTER = []
        PRED = []
        REAL = []
        for pred, real in zip(pred_list, real_list):
            inter = [i for i in pred if i in real]
            INTER.append(len(inter))
            PRED.append(len(pred))
            REAL.append(len(real))
        INTER = float(sum(INTER))
        PRED = float(sum(PRED))
        REAL = float(sum(REAL))
        hP = INTER / PRED
        hR = INTER / REAL
        Fn = ((n * n + 1) * hP * hR) / (n * n * hP + hR)

        count_all = 0
        count = 0
        for pred, real in zip(pred_list, real_list):
            if pred[1]==real[1]:
                count +=1
            count_all +=1
        acc = count/count_all

        return hP, hR, Fn,acc


def assessment(y_test, y_pred):
    '''Accuracy'''
    print(accuracy_score(y_true=y_test, y_pred=y_pred))
    '''Classification report'''
    print(classification_report(y_true=y_test,y_pred=y_pred,digits=3))
    '''confusion metrics'''
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(cm)
    return cm


def plot(pred_class):
	global Y_01234
	# take inputs of Y_test and pred_class to calculate metrics
	Y_test = Y_01234
	y_pred_class = pred_class
	n_classes = np.max(Y_test) + 1
	classes = np.arange(n_classes).tolist()
	# accuracy
	try:
	    Y_test = Y_test[:,0]
	except:
	    pass
	accuracy = accuracy_score(y_true=Y_test, y_pred=y_pred_class)
	print("Accuracy: " + str(accuracy) + '\n')
	print("Classification report:\n")
	# print metrics
	print(classification_report(y_true=Y_test,
	                            y_pred=y_pred_class,
	                            ))
	# print confusion metrics
	print("Confusion metrics:\n")
	confusion = confusion_matrix(y_true=Y_test, y_pred=y_pred_class)
	name_list = ['real class 0', 'real class 1', 'real class 2', 'real class 3', 'real class 4']

	x = np.arange(len(name_list))
	total_width, n = 0.8, 5
	width = total_width / n
	# x = x - (total_width - width) / 2

	plt.bar(x, confusion[:, 0], label='predict class 0', width=width)
	plt.bar(x + width, confusion[:, 1], label='predict class 1', width=width)
	plt.bar(x + width * 2, confusion[:, 2], label='predict class 2', tick_label=name_list, width=width)
	plt.bar(x + width * 3, confusion[:, 3], label='predict class 3', width=width)
	plt.bar(x + width * 4, confusion[:, 4], label='predict class 4', width=width)

	# plt.
	plt.legend()
	plt.show()



if __name__ == '__main__':
	real_list = list()
	Y_01234 = list()

	# init classfiers
	clf_01234 = ClassifierAdaptor('../8w_01234_classifier.joblib')
	clf_zk0 = ClassifierAdaptor('../8w_zk0_classifier.joblib')
	clf_zk1 = ClassifierAdaptor('../8w_zk1_classifier.joblib')
	clf_zk2 = ClassifierAdaptor('../8w_zk2_classifier.joblib')
	clf_zk3 = ClassifierAdaptor('../8w_zk3_classifier.joblib')
	clf_zk4 = ClassifierAdaptor('../8w_zk4_classifier.joblib')

	print("================")

	# init learner
	hierarchical_clf = HierarchicalLearning()
	hierarchical_clf.set_root_classifier(clf_01234)
	second_layer_clfs = list([clf_zk0, clf_zk1, clf_zk2, clf_zk3, clf_zk4])
	hierarchical_clf.set_sub_classifiers(second_layer_clfs)
	
	test_list = list(testset_instance_labels.keys())
	X_test, Y_test = get_test_instance_vector_label(test_list)

	total_count = 0
	correct_count = 0
	pred_results = []

	start_time = time.time()
	for i in range(0, len(X_test)):	
		total_count += 1
		y_pred = hierarchical_clf.classify(X_test[i])
		# print("[INFO] true:" + str(y_pred) + " | pred:" + str(Y_test[i]))
		pred_results.append(y_pred)
		if y_pred == Y_test[i]:
			correct_count += 1
	end_time = time.time()

	print("Accuracy : " + str(correct_count / total_count))

	hP, hR, Fn,acc = hierarchical_acc(hierarchical_clf.pred_list, real_list, 1)
	print("hp : " + str(hP))
	print("hr : " + str(hR))
	print("fn : " + str(Fn))
	print("acc : " + str(acc))

	print("==============================")
	Y_pred_col = np.array(pred_results).reshape(-1, 1)
	Y_real_col = np.array(Y_test).reshape(-1, 1)
	assessment(Y_real_col, Y_pred_col)
	print("==============================")
	print(str(end_time - start_time) + "s")
	print(hierarchical_clf.time)

	plot(hierarchical_clf.result_01234)