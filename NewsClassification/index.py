import os
import sys
import jieba
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import model_util


def read_file(path):
    fp = open(path, 'r', encoding='UTF-8')
    try:
        content = fp.read()
        fp.close()
        return content
    except OSError:
        return ""


def segment(content):
    if content == '':
        return []
    content = content.replace(",", "").replace("\n", "").replace("\r", "").replace("\n", "")
    return " ".join(jieba.cut(content))


def get_segment_data(path):
    source_path = path + '/' + 'source'
    if os.path.isfile(path + '/source.csv') == True:
        return pd.DataFrame.from_csv(path + '/source.csv')
    dict_data = {'tag': [], 'data': []}
    catelist = os.listdir(source_path)
    i = 1
    time1 = time.time()
    for val in catelist:
        i += 1
        if (i % 100) == 0:
            print('进度：%.2f%% 用时：%ds' % (i / len(catelist) * 100, time.time() - time1))
        file_path = source_path + '/' + val
        if os.path.isfile(file_path) == False:
            continue
        content = read_file(file_path)
        if content == '':
            continue

        tag = val.split('_')
        dict_data['tag'].append(tag[0])
        dict_data['data'].append(segment(content))

    df = pd.DataFrame.from_dict(dict_data)
    df.to_csv('./data/source.csv', encoding='utf-8')
    return df


def get_tfid_data(path):
    if os.path.isfile(path + '/tfidf.bin') == True:
        fp = open(path + '/tfidf.bin', 'rb')
        data = pickle.load(fp)
        fp.close()
        return data
    stop_words = read_file('./data/stop_words.txt').splitlines()
    data = get_segment_data(path)
    vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, max_df=0.8, max_features=10000)
    vect = vectorizer.fit_transform(data['data'])
    tag = data['tag']
    vect_data = {'data': vect, 'tag': tag}
    fp = open(path + '/tfidf.bin', 'w+b')
    pickle.dump(vect_data, fp)
    fp.close()
    return vect_data


data_path = './data'
vect_data = get_tfid_data(data_path)

x_train, x_test, y_train, y_test = train_test_split(vect_data['data'], vect_data['tag'], test_size=0.25, random_state=0)




model_util.lr_model(x_train, y_train, x_test, y_test, 'l2', 1)
model_util.nn_model(x_train, y_train, x_test, y_test, alpha=0.01)
model_util.rd_model(x_train, y_train, x_test, y_test, 50, 20)
