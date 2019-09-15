# classificationOfResidentialRequests

Implementation of Hybrid Machine Learning Models of Classifying Residential Requests in Smart Cities Paper
<a href="https://github.com/OneClickDeepLearning/classificationOfResidentialRequests/blob/master/Hybrid%20Machine%20Learning%20Models%20of%20Classifying%20Residential%20Requests%20for%20Smart%20Dispatching.pdf">Our Paper</a>

## Local Environment Specification
Training and testing ran on a machine with:
- Ubuntu 16.04 LTS
- Nvidia GeForce GTX 1070
- CUDA version: 9.0
- Cudnn version: 7.3.0
- Python version: 3.5.2
- Tensorflow-gpu: 1.11.0
- Keras: 2.2.4

## Introduction
This implementation includes all the tasks that was described in the paper, including feature engineering, hybrid machine learning, different classifiers, convolution neural network models, etc. We split the implementation in to four parts:
- Bayesian model
- Neural network model
- Feature engineering
 
## <a href="https://github.com/OneClickDeepLearning/classificationOfResidentialRequests/tree/master/feature_engineering">Feature Engineering</a>
Feature engineering processes and transforms the data set in Chinese texts to word vectors as inputs of machine learning models. 
- Data Preprocess
	- Segmented into tokens
	- Remove punctuation, stopwords, etc.
- Lexical Analysis (request, category, responsible department description)
	- Data Distribution
	- Information Values of Features
- Word Embedding and Vectorization
	- Word embedding using Word2Vec 
	- Word vector using TF-IDF
## <a href="https://github.com/OneClickDeepLearning/classificationOfResidentialRequests/tree/master/bayes">Hierarchical Classification</a>
We develop a hierarchical classification method to handle classification.
- K-Means and GMM Clustering
- OPTICS, LDA and Entropy Calculation
## Hybrid Machine Learning Models
- Bayesian classifier
- Hierarchical Bayesian classifier
- Fully-connected NN classifier
- Hierarchical fully-connected NN classifier
- Residual convolutional NN classifier
## Performance on blind test set
<table>
    <tr>
        <td rowspan="3">Models</td>
        <td colspan="5">Metrics <br></td>
    </tr>
    <tr>
        <td rowspan="2">Accuracy</td>
        <td colspan="2">Precision</td>
        <td colspan="2">Recall</td>
    </tr>
    <tr>
        <td>Micro</td>
        <td>Macro</td>
        <td>Micro</td>
        <td>Macro</td>
    </tr>
    <tr>
        <td>Hierarchical Fully Connected NN</td>
        <td>0.6495</td>
        <td>0.650</td>
        <td>0.244</td>
        <td>0.650 <br></td>
        <td>0.192 <br></td>
    </tr>
    <tr>
        <td>Fully Connected NN</td>
        <td>0.6889</td>
        <td>0.689 <br></td>
        <td>0.259 <br></td>
        <td>0.689 <br></td>
        <td>0.214 <br></td>
    </tr>
    <tr>
        <td>Hierarchical Naive Bayesian</td>
        <td>0.6776</td>
        <td>0.678 <br></td>
        <td>0.251 <br></td>
        <td>0.678 <br></td>
        <td>0.201 <br></td>
    </tr>
    <tr>
        <td>Naive Bayesian</td>
        <td>0.7258 <br></td>
        <td>0.726 <br></td>
        <td>0.295 <br></td>
        <td>0.726 <br></td>
        <td>0.256 <br></td>
    </tr>
    <tr>
        <td>Residual Network</td>
        <td>0.7642</td>
        <td>0.764 <br></td>
        <td>0.417 <br></td>
        <td>0.764 <br></td>
        <td>0.352 <br></td>
    </tr>
</table>
