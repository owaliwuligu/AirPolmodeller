#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn.linear_model import LinearRegression
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from xgboost import plot_importance
from script import validation
from script import process_str
from ml import xgb
from ml import linear
from ml import svm
from ml import decision_tree
from ml import knn
from ml import gradient_boosting
from ml import random_forest
from ml import extra_tree
from ml import lasso_glm

class bagging_predictor:
    def __init__(self, model_num, model_list, predictor_type):
        self.model_num = model_num
        self.model_list = model_list
        self.predictor_type = predictor_type
    def predict(self, X_predict):
        Y_predict = []
        if self.predictor_type=='regression':
            for i in range(len(X_predict)):
                Y_predict.append(0.0)
            for p in self.model_list:
                Y_temp = p.predict(X_predict)
                for i in range(len(Y_temp)):
                    Y_predict[i] += Y_temp[i]
            for i in range(len(X_predict)):
                Y_predict[i] /= self.model_num
            return Y_predict
        else:
            for x in X_predict:
                x_temp = []
                x_temp = x_temp.append(x)
                label_count = {}
                for p in self.model_list:
                    y_temp = p.predict(x_temp)
                    if str(y_temp) not in label_count.keys():
                        label_count[str(y_temp)] = 1
                    else:
                        label_count[str(y_temp)] += 1
                max_id = 0
                max_value = 0
                for key in label_count.keys():
                    if label_count[key]>max_value:
                        max_value = label_count[key]
                        max_id = int(key)
                Y_predict.append(max_id)
            return Y_predict

def train(X_train, Y_train, predictor_type, method_param_list):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define params for each algorithm
    #xgboost
    param_value_xgboost = {
    'booster':'gbtree',
    'max_depth':6,
    'gamma':0,
    'eta':0.3,
    'objective':'reg:gamma',
    'silent':False,
    'learning_rate':0.1,
    'n_estimators':160,
    'min_child_weight':1,
    'max_delta_step':0,
    'subsample':1,
    'colsample_bytree':1,
    'colsample_bylevel':1
    }
    #linear
    param_value_linear = {
    'fit_intercept':True,
    'normalize':False,
    'copy_X':True,
    'n_jobs':1
    }

    model_list = []

    for mp_key in method_param_list.keys():
        real_mp_key = mp_key.split('-')[0]
        if real_mp_key=='xgboost':    
            predictor = xgb.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='linear':
            predictor = linear.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='svm':
            predictor = svm.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='decision_tree':
            predictor = decision_tree.train(X_train, Y_train, predictor_type,method_param_list[mp_key])
        elif real_mp_key=='knn':
            predictor = knn.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='gradient_boosting':
            predictor = gradient_boosting.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='random_forest':
            predictor = random_forest.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='extra_tree':
            predictor = extra_tree.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif real_mp_key=='lasso_glm':
            predictor = lasso_glm.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        elif reall_mp_key=='mlp':
            predictor = mlp.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
        model_list.append(predictor)
    bp = bagging_predictor(len(model_list), model_list, predictor_type)
#    print(bp.model_num)
#    print(len(model_list))
    return bp
