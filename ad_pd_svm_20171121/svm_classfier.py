import pandas as pd
import numpy as np
import sklearn
import os
import sys
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV


ad_path = '/home/qwerty/workspcae/ad_pd_svm_20171121/data/ad_res/'
pd_path = '/home/qwerty/workspcae/ad_pd_svm_20171121/data/pd_csv_txt/'
nor_path = '/home/qwerty/workspcae/ad_pd_svm_20171121/data/nor_res/'

def load_data(path, flag=None):
    df = pd.DataFrame()
    p_data_set = {}
    for parent, dirnames, filenames in os.walk(path):
        if filenames:
            for filename in filenames:
                filepath = parent + '/' + filename

                p_data = {}
                with open(filepath) as f:
                    if parent[-3:] == "csv":
                        lables, data_set = (i.strip().split('\t') for i in f.readlines())
                        assert len(lables) == len(data_set)
                        for i in range(1, len(lables)):
                            p_data[lables[i]] = float(data_set[i])
                        p_id = parent.split('/')[-2].split('_')[0] + '_' + data_set[0].split('_')[-1]
                    else:
                        for line in f:
                            if line:
                                lable, data = line.strip().split(' ')
                                lable = filename.split('.')[0].split('_')[1] + lable
                                p_data[lable] = float(data)
                        p_id = parent.split('/')[-2].split('_')[0] + '_' + filename.split('_')[0]
                if p_id in p_data_set.keys():
                    p_data_set[p_id].update(p_data)
                else:
                    p_data_set[p_id] = p_data
    for k, v in p_data_set.items():
        df[k] = pd.Series(data=v)
    return df.T

def svm_cross_validation(x_train, y_train, model, c_list=[1], gamma_list=['auto']):
    param_grid = {'C': c_list, 'gamma': gamma_list}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
    grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(x_train, y_train)
    return model


if __name__ == "__main__":
    ad_df = load_data(ad_path)
    pd_df = load_data(pd_path)
    nor_df = load_data(nor_path)
    linear_clf = svm.SVC(kernel='linear', probability=True)
    poly_clf = svm.SVC(kernel='poly')
    rbf_clf = svm.SVC(kernel='rbf')
    c_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    gamma_list = [1e-3, 1e-2, 1e-1, 1, 10]
    x = []
    y = []
    for p in pd_df.index:
        x.append(pd_df.loc[p].values)
        y.append(1)
    for p in nor_df.index:
        x.append(nor_df.loc[p].values)
        y.append(-1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model = svm_cross_validation(x_train, y_train, linear_clf, c_list)
    y_predict = model.predict(x_test)
    acc = 1 - np.count_nonzero((y_test - y_predict)) / float(len(y_predict))
    print(y_predict, y_test, acc)
    # scores = cross_val_score(linear_clf, x_train, y_train, cv=3)
    # print(scores)


# [ 0.6557377   0.72131148  0.65        0.69491525  0.71186441] linear kernel
# [ 0.52459016  0.52459016  0.53333333  0.52542373  0.52542373] rbf_kernel
