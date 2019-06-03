import os
import csv
import json
import re
import traceback
import pickle
import requests
from bs4 import BeautifulSoup



file_path = "data/4k_testset_01234.csv"
output_file_prefix = "test01234/testset_"


source_file = open(file_path, 'r', encoding='gb18030')
reader = csv.reader(source_file)
regexp_number = re.compile('^[0-9a-zA-Z]*$')
regexp_punct = re.compile("^[\s+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）. 〉《 》〔〕；～ ％]$")
stop_words_lst = ["市民", "来电", "咨询", "反映", "职能", "规定", "局", "内容", "工单", "问题"]
NI_suffix_tuple = ("局", "队", "所", "会", "中心", "部门")


instance_labels = dict()
instance_tokens = dict()
instance_classes = dict()  # here is diff with trainset(class_index), here is real class


for row in reader:
	instance_id = row[0]
	label = row[9]
	content = row[6]

	class_1 = row[2]
	class_2 = row[3]
	class_3 = row[4]
	class_4 = row[5]

	if instance_id.strip() == "ID":
		continue

	# collect classes
	class_labels = []
	class_labels.append(class_1)
	class_labels.append(class_2)
	class_labels.append(class_3)
	class_labels.append(class_4)

	# build request
	payload = dict()
	payload['s'] = content
	payload['f'] = 'xml'
	payload['t'] = 'ner'
	response = requests.post("http://127.0.0.1:12345/ltp", data=payload, timeout=5)
	soup = BeautifulSoup(response.text, 'html.parser')
	word_tags = soup.findAll('word')
	# parse and extract features
	buffers = list()
	tokens = list()

	for w in word_tags:
		token = w['cont']
		pos = w['pos']
		ner = w['ne']

		# rm stop words
		if token in stop_words_lst and ner is 'O':
			continue

		# merge continuous nouns
		if ner.startswith('B-') or ner.startswith('I-'):
			buffers.append(token)
			continue

		if ner.startswith('E-'):
			buffers.append(token)
			token = ''.join(buffers)
			buffers.clear()

		# note the NER
		if ner is not 'O':
			pos = ner[-2:]

		# filter numbers & punct
		if regexp_number.match(token.strip()):
			print("[INFO] invalid token : alphnum")
			continue

		if regexp_punct.match(token.strip()) or pos == 'wp':
			print("[INFO] invalid token : punctuation")
			continue

		# custom rules
		if (pos == 'j' or pos == 'n') and len(token) >=3 and token.endswith(NI_suffix_tuple):
			pos = 'Ni'
			print("[INFO] pos of token should be Ni : " + token)

		# build inverse index
		if pos not in ('Ni', 'n', 'j'):  # Ns exclusive
			print("[INFO] exclusive :" + token + ":" + pos)   # print, but throw out

		else:
			tokens.append(token)
			print("[INFO] add :" + token)

	instance_labels[instance_id] = label
	instance_tokens[instance_id] = tokens
	instance_classes[instance_id] = class_labels
	print("-----------------------------")


print("================ END =================")
print("total instance :" + str(len(list(instance_tokens.keys()))))


with open(output_file_prefix + 'bow_instance_label.pickle', 'wb') as f:
	pickle.dump(instance_labels, f, pickle.HIGHEST_PROTOCOL)

with open(output_file_prefix + 'bow_instance_tokens.pickle', 'wb') as f:
	pickle.dump(instance_tokens, f, pickle.HIGHEST_PROTOCOL)

with open(output_file_prefix + 'bow_instance_classes.pickle', 'wb') as f:
	pickle.dump(instance_classes, f, pickle.HIGHEST_PROTOCOL)