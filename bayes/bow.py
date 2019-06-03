import os
import csv
import json
import re
import traceback
import pickle
import requests
from bs4 import BeautifulSoup


global inverse_index
global testset_instance_tokens

# 8w_all
with open("data/8w_zkall/trainset_bow_inverse_index.pickle", 'rb') as iif:
	inverse_index = pickle.load(iif)

with open("data/4k_test/testset_bow_instance_tokens.pickle", 'rb') as itf:
	testset_instance_tokens = pickle.load(itf)


def get_bag_of_word_vector(instance_id):
	vector = []
	for token, iid_set in inverse_index.items():
		if instance_id in iid_set:
			vector.append(1)
		else:
			vector.append(0)

	return vector



def get_testset_bow_vector(instance_id):
	if instance_id not in testset_instance_tokens.keys():
		print("[ERROR] instance id is not invalid")
	else:	
		tokens = testset_instance_tokens[instance_id]
		dimension = list(inverse_index.keys())
		vector = []
		for d in dimension:
			if d in tokens:
				vector.append(1)
			else:
				vector.append(0)

		return vector