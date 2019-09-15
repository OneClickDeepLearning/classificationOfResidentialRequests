# Feature Engineering 

## Required packages
```
$ pip install -r requirements.txt
```

## Introduction
This folder is feature engineering. 
There are two parts, first one is feature analysis based on IV calculation.
The other is clustering algorithm, including KMEANS, GMM, DBSCAN, LDA and others.

## Usage
`cluster_algo` has all clustering algorithm mentioned before. 
To use this function, you need generate your own `agency_to_vec.pickle` or other files, 
which including word to vector. For `place_cluserting.py`, it is the implementation for Topic-based Clustering mentioned in paper.
For `dbscan.py`, it is another implementation for clustering, for us to analyze the data better. 

