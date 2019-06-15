# classificationOfResidentialRequests

Implementation of Hybrid Machine Learning Models of Classifying Residential Requests in Smart Cities Paper
> Link of our paper

## Introduction
This implementation includes all the tasks that was described in the paper, including feature engineering, hybrid machine learning, different classifiers, convolution neural network models, etc. We split the implementation in to four parts:
- Bayesian model
- Neural network model
- Feature engineering
 
## Feature Engineering
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
## Hierarchical Classification 
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
