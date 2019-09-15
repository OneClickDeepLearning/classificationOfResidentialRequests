# Hierarchical bayes model
## Introduction
Hierarchical baye model consists of 1 model as the first layer, and 5 model as the second layer. The First layer model(4w_01234_classifier) is to classify the input into 5 classes(zk0, zk1, zk2, zk3, zk4), and then the corresponding model in the second layer will used to do the further classification(4w_zk0_classifier, 4w_zk1_classifier, 4w_zk2_classifier, 4w_zk3_classifier, 4w_zk4_classifier).

Here, we trained the Hierarchical bayes model over 40,000, and 80,000 dataset respectively. The corresponding model dump file as follow:
```
4w_01234_classifier
	|- 4w_zk0_classifier 
	|- 4w_zk1_classifier
	|- 4w_zk2_classifier
	|- 4w_zk3_classifier
	|- 4w_zk4_classifier 

8w_01234_classifier
	|- 8w_zk0_classifier 
	|- 8w_zk1_classifier
	|- 8w_zk2_classifier
	|- 8w_zk3_classifier
	|- 8w_zk4_classifier
```

## Processing && Reproduction
1. Labeling 
* Groups training dataset, preparing dataset to train the differernt models in hierarchical tree.
* To train the first layer model, all training data should under the label of (0,1,2,3,4).
* To train the second layer models, XX_zk0_classifier for example, all training dataset with other group labels need to be filtered out.
* Relative code: <a href="https://github.com/Tann-chen/classificationOfResidentialRequests/tree/master/bayes/label">./label</a>
2. Data pre-processing
* Generates inverse-index, all instance labels list, instance classes list to be used in training process.
- Relative code: process_testset.py, process_trainset.py, process_blind_test.py
- <a href="https://github.com/HIT-SCIR/ltp">LTP platform</a> is used to NER, tokenize the chinese words.
* To train classifiers in hierarchical tree, the generated data structure will be used from previous step.
- Relative code: sklearn_bayes.py
- Generated models: XX_zkXX_classifier.joblib
3. Applying hierarchical models & testing
* Relative code: <a href="https://github.com/Tann-chen/classificationOfResidentialRequests/tree/master/bayes/hierarchical_bayes">./hierarchical_bayes</a>
- main.py is the entry of code
