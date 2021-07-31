#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
import xgboost as xgb
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from xgboost import plot_importance
from script import validation
from script import process_str
from sklearn.multiclass import OneVsRestClassifier
def train(X_train, Y_train, predictor_type, param_list, OneVsRest=False):
    #judge the type of model: classication or regression
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define the params
    param_value_re = {
    'booster':'gbtree',
    'max_depth':6,
    'gamma':0,
    'eta':0.3,
    'objective':'reg:squarederror',
    'silent':False,
    'learning_rate':0.1,
    'n_estimators':160,
    'min_child_weight':1,
    'max_delta_step':0,
    'subsample':1,
    'colsample_bytree':1,
    'colsample_bylevel':1
    }

    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='booster':
            if param_list[key]!='gbtree' and param_list[key]!='gblinear':
                print('The value of the parameter \'booster\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \'booster\' is not valid!\n')
                sys.exit(1)
            param_value_re[key] = param_list[key]
        elif key=='objective':
            if param_list[key] not in ['reg:linear', 'reg:logistic', 'reg:gamma', 'binary:logistic', 'binary:logitraw', 'count:poisson', 'multi:softmax', 'multi:softprob', 'rank:pairwise']:
                print('The value of the parameter \'objective\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \'objective\' is not valid!\n')
                sys.exit(1)
            param_value_re[key] = param_list[key]
        elif key=='silent':
            if param_list[key]!=True and param_list[key]!=False and process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \'silent\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \'silent\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        else:
            if process_str.check_letter(str(param_list[key]))==False:
                if key in ['max_depth', 'silent', 'n_estimators', 'max_delta_step']:
                    param_value_re[key] = int(param_list[key])
                else:
                    param_value_re[key] = param_list[key]
            else:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
    if predictor_type=='regression':
#        clf = xgb.XGBRegressor(objective='reg:gamma')
#        predictor = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
        predictor = xgb.XGBRegressor(booster=param_value_re['booster'], max_depth = param_value_re['max_depth'], gamma=param_value_re['gamma'], eta=param_value_re['eta'], objective=param_value_re['objective'], slient=param_value_re['silent'], learning_rate=param_value_re['learning_rate'], n_estimators=param_value_re['n_estimators'], min_child_weight=param_value_re['min_child_weight'], max_delta_step=param_value_re['max_delta_step'], subsample=param_value_re['subsample'], colsample_bytree=param_value_re['colsample_bytree'])
    else:
        predictor = xgb.XGBClassifier(booster=param_value_re['booster'], max_depth = param_value_re['max_depth'], gamma=param_value_re['gamma'], eta=param_value_re['eta'], objective=param_value_re['objective'], slient=param_value_re['silent'], learning_rate=param_value_re['learning_rate'], n_estimators=param_value_re['n_estimators'], min_child_weight=param_value_re['min_child_weight'], max_delta_step=param_value_re['max_delta_step'], subsample=param_value_re['subsample'], colsample_bytree=param_value_re['colsample_bytree'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor

def predict(X_predict, model_path):
    X_predict = np.array(X_predict)
    f = open(model_path, 'rb')
    predictor = pickle.load(f)
    f.close()
    Y_predict = predictor.predict(X_predict)

    return Y_predict    
