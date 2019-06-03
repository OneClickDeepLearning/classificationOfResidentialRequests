import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# sents = [
# 'I am a good student'.split(),
# 'Good good study day day up'.split()
# ]
# model = word2vec.word2vec(sents, size=100, window=5, min_count=2, workers=10)
# # 打印单词'good'的词向量
# print(model.wv.word_vec('good'))
# # 打印和'good'相似的前2个单词
# print(model.wv.most_similar('good', topn=2))
# # 保存模型到文件
# model.save('w2v.model')


file1 = '../data/split_data.txt'
file2 = '../models/CBOW.model'
model = Word2Vec(LineSentence(file1),sg=0,size = 100, window= 5,workers=multiprocessing.cpu_count())
model.save(file2)