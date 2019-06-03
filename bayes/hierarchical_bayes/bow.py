import os
import csv
import json
import re
import traceback
import pickle
import requests
from bs4 import BeautifulSoup

with open("../data/8w_01234/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_01234 = pickle.load(iif)

with open("../data/8w_zk0/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_zk0 = pickle.load(iif)

with open("../data/8w_zk1/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_zk1 = pickle.load(iif)

with open("../data/8w_zk2/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_zk2 = pickle.load(iif)

with open("../data/8w_zk3/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_zk3 = pickle.load(iif)

with open("../data/8w_zk4/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index_zk4 = pickle.load(iif)

with open("../data/blind_test/testset_bow_instance_tokens.pickle", 'rb') as itf:
	testset_instance_tokens = pickle.load(itf)



def get_testset_bow_vector(instance_id, node_id):
	global inverse_index_01234
	global inverse_index_zk0
	global inverse_index_zk1
	global inverse_index_zk2
	global inverse_index_zk3
	global inverse_index_zk4
	global testset_instance_tokens

	if instance_id not in testset_instance_tokens.keys():
		print("[ERROR] instance id is not invalid")
	else:	
		tokens = testset_instance_tokens[instance_id]
		if node_id == '01234':
			inverse_index = inverse_index_01234
		elif node_id == 'zk0':
			inverse_index = inverse_index_zk0
		elif node_id == 'zk1':
			inverse_index = inverse_index_zk1
		elif node_id == 'zk2':
			inverse_index = inverse_index_zk2
		elif node_id == 'zk3':
			inverse_index = inverse_index_zk3
		elif node_id == 'zk4':
			inverse_index = inverse_index_zk4

		dimension = list(inverse_index.keys())

		vector = []
		for d in dimension:
			if d in tokens:
				vector.append(1)
			else:
				vector.append(0)

		return vector