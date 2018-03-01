import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


def lr_model(train_data, train_label, test_data, test_label, penalty, C):
    md = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty=penalty, C=C)
    return _performance(md, train_data, train_label, test_data, test_label)


def svm_model(train_data, train_label, test_data, test_label):
    md = SVC(kernel='rbf')
    return _performance(md, train_data, train_label, test_data, test_label)


def bayes_model(train_data, train_label, test_data, test_label):
    md = GaussianNB()
    return _performance(md, train_data, train_label, test_data, test_label)


def gbdt_model(train_data, train_label, test_data, test_label):
    md = GradientBoostingClassifier()
    return _performance(md, train_data, train_label, test_data, test_label)


def rd_model(train_data, train_label, test_data, test_label, n, depth):
    md = RandomForestClassifier(n_estimators=n, max_depth=depth)
    return _performance(md, train_data, train_label, test_data, test_label)


def ada_model(train_data, train_label, test_data, test_label, n):
    md = AdaBoostClassifier(n_estimators=n)
    return _performance(md, train_data, train_label, test_data, test_label)


def nn_model(train_data, train_label, test_data, test_label, hidden_layer_sizes=4, alpha=0.0001):
    md = MLPClassifier(hidden_layer_sizes, activation='relu', alpha=alpha)
    return _performance(md, train_data, train_label, test_data, test_label)


def _performance(md, train_data, train_label, test_data, test_label):
    fit_begin = time.time()
    md.fit(train_data, train_label)
    fit_time = time.time() - fit_begin

    train_score = md.score(train_data, train_label)
    test_score = md.score(test_data, test_label)
    test_pred_begin = time.time()
    train_pred = md.predict(train_data)
    test_pred = md.predict(test_data)
    test_pred_time = (time.time() - test_pred_begin) / len(test_label)

    print("训练报告：======>")
    print(str(md))
    print("训练样本数：" + str(len(train_label)) + " 测试样本数：" + str(len(test_label)))
    print('训练时间：' + str(fit_time) + '秒  预测单个样本时间：' + str(test_pred_time) + '秒')
    print('训练集分数：' + str(train_score) + ' 测试集分数：' + str(test_score))
    print("<======")
    return train_pred, test_pred, train_score, test_score

