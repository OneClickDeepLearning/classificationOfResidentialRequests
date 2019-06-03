import pandas as pd
import time
import re
import requests
from bs4 import BeautifulSoup
import csv
import re
import jieba
import numpy as np
from gensim.models import Word2Vec


def further_clean():
    labels = {}
    label_file = '../data/labels_merge.txt'
    for line in open(label_file,'r',encoding='utf-8'):
        try:
            labels[line.split(',')[0]] = line.split(' ')[1]
        except:
            continue

    return labels

def hgdProcess_dept(sentence):
    labels = further_clean()
    payload = {}
    payload['s'] = sentence
    payload['f'] = 'xml'
    payload['t'] = 'pos'
    ns = []
    response = requests.post("http://127.0.0.1:12345/ltp", data=payload)
    # docker run -d -p 12345:12345 ce0140dae4c0 ./ltp_server --last-stage all
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup)
    word_tags = soup.findAll('word')
    # for word in word_tags:
    #     word = word['cont']
    for word in word_tags:
        if (word['pos'] == 'ns'):
            if (word['cont'] != "中国"):
                sentence = re.sub(word['cont'], '', sentence)
                ns.append(word['cont'])
            if (re.sub("海口", '', word['cont']) != word['cont']):
                ns.clear()
                ns.append("海口市")
    if (len(sentence) > 2 and sentence[2] == "区"):
        ns.append(sentence[0:3])
        sentence = re.sub(sentence[0:3], "", sentence)
    if (len(sentence) > 2 and re.sub("街道办", '', sentence) != sentence):
        diming = re.sub("街道办", '', sentence)
        if (diming != ''):
            ns.append(diming)
        sentence = re.sub(diming, '', sentence)
    if (len(ns) == 0 and sentence[1]!="委" and sentence[1]!="政" ):
        sentence = re.sub('市', "", sentence)
    sentence = re.sub('分公司', "", sentence)
    if (sentence == "政府" or sentence == "镇政府"):
        sentence = "人民政府"
    elif (re.sub("镇政府", "", sentence) != sentence):
        sentence = "人民政府"
        ns.append(re.sub("镇政府", "", sentence))
    elif (sentence == "委办" or sentence == "委"):
        sentence = "市委办公室"
    elif (sentence == "片区棚户区（）改造项目指挥部"):
        sentence = "棚户区改造项目指挥部"
    elif (re.sub("棚改指挥部","",sentence)!=sentence):
        sentence = "棚户区改造项目指挥部"
        ns.append(re.sub("棚改指挥部","",sentence))
    elif (sentence == "纪委" or sentence == "纪委(监察局)"):
        sentence = "纪委监察局"
    # elif (sentence == "省外单位" or sentence == "无效归属"):
    #     sentence = "无效数据"
    elif (sentence == "面前坡片区改造项目指挥部"):
        sentence = "改造项目指挥部"
        ns.append('面前坡片区')
    elif(re.sub("组织部","",sentence)!=sentence):
        sentence = "组织部"
    elif(sentence=='中国国民党革命委员会委员会'):
        sentence="中国国民党革命委员会"
    elif(sentence=='残联'):
        sentence='残疾人联合会'
    elif(sentence=='城管局'):
        sentence='城市管理行政执法局'
    elif(sentence=='住建局'):
        sentence="住房和城乡建设局"
    elif(sentence=='物价局'):
        sentence='物价监督局'
    elif (sentence == '园林局'):
        sentence = '园林管理局'
    elif (sentence == '国资'):
        sentence = '国土资源局'
    elif (sentence == '人社局'):
        sentence = '人力资源和社会保障局'
    elif (sentence == '科工信局'):
        sentence = '科学技术工业信息化局'

    try:
        sentence = re.sub('\n','',labels[sentence])
    except:
        sentence = sentence
    return sentence


def jieba_process_content(cont,model):


    cont = re.sub('市民来电咨询', '', cont)
    cont = re.sub('市民来电反映', '', cont)
    cont = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[a-zA-Z0-9+——！，。？、~@#￥%……&*（）《》：:]+", "", cont)
    splits = jieba.cut(cont)

    vector = np.array([])
    for word in splits:
        try:
            vec = model[word]
            vector = np.append(vector, vec)
            if (vector.shape[0] == 10000):
                break
        except Exception:
            continue

    if (vector.shape[0] < 10000):
        pendding = np.zeros(10000 - vector.shape[0])
        vector = np.append(vector, pendding)

    return vector

label = further_clean()
print(label)