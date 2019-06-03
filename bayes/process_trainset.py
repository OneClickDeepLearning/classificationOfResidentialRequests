import os
import csv
import json
import re
import traceback
import pickle
import requests
from bs4 import BeautifulSoup


def get_class_label(class_name, class_lst):
	index = -1
	for c in class_lst:
		index = index + 1
		if c == class_name:
			return index
	# if never found
	class_lst.append(class_name)
	return index + 1


def processing(which):
	file_path = "data/4w_trainset_" + which + ".csv"
	output_file_prefix = "data/4w_" + which + "/trainset_"

	os.mkdir("data/4w_" + which)

	source_file = open(file_path, 'r', encoding='gb18030')
	reader = csv.reader(source_file)

	regexp_number = re.compile('^[0-9a-zA-Z]*$')
	regexp_punct = re.compile("^[\s+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）. 〉《 》〔〕；～ ％]$")
	stop_words_lst = ["市民", "来电", "咨询", "反映", "职能", "规定", "局", "内容", "工单", "问题"]
	NI_suffix_tuple = ("局", "队", "所", "会", "中心", "部门")


	inverse_index = dict()
	instance_labels = dict()
	instance_tokens = dict()
	instance_classes = dict()

	first_class_lst = list()
	second_class_lst = list()
	third_class_lst = list()
	fourth_class_lst = list()

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

		class_1_label = get_class_label(class_1, first_class_lst)
		class_labels.append(class_1_label)

		class_2_label = get_class_label(class_2, second_class_lst)
		class_labels.append(class_2_label)

		class_3_label = get_class_label(class_3, third_class_lst)
		class_labels.append(class_3_label)

		class_4_label = get_class_label(class_4, fourth_class_lst)
		class_labels.append(class_4_label)


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

			elif token in inverse_index.keys():
				instances_set = inverse_index[token]
				instances_set.add(instance_id)
				tokens.append(token)
				print("[INFO] add :" + token)
			else:
				new_set = set()
				new_set.add(instance_id)
				inverse_index[token] = new_set
				tokens.append(token)
				print("[INFO] add :" + token)

		instance_labels[instance_id] = label
		instance_tokens[instance_id] = tokens
		instance_classes[instance_id] = class_labels
		print("-----------------------------")


	print("================ END =================")
	print("totoal dimension :" + str(len(list(inverse_index.keys()))))
	print("total instance :" + str(len(list(instance_tokens.keys()))))


	with open(output_file_prefix + 'bow_inverse_index.pickle', 'wb') as f:
		pickle.dump(inverse_index, f, pickle.HIGHEST_PROTOCOL)

	with open(output_file_prefix + 'bow_instance_label.pickle', 'wb') as f:
		pickle.dump(instance_labels, f, pickle.HIGHEST_PROTOCOL)

	with open(output_file_prefix + 'bow_instance_classes.pickle', 'wb') as f:
		pickle.dump(instance_classes, f, pickle.HIGHEST_PROTOCOL)

	# dump classes list
	with open(output_file_prefix + 'bow_classes_list.pickle', 'wb') as f:
		classes_list = list()
		classes_list.append(first_class_lst)
		classes_list.append(second_class_lst)
		classes_list.append(third_class_lst)
		classes_list.append(fourth_class_lst)
		pickle.dump(classes_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	classes = ["01234", "zk0", "zk1", "zk2", "zk3", "zk4"]
	qzk = ["zkall"]

	for i in qzk:
		processing(i)