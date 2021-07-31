#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pickle
from script import validation
from script import process_str

def train(X_train, Y_train, predictor_type, param_list, OneVsRest=False):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define the params
    param_value_re = {
    'n_neighbors':5,
    'weights':'uniform',
    'p':2,
    'algorithm':'auto'
    }
    
    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='weights':
            if param_list[key] not in ['uniform', 'distance']:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = param_list[key]
        elif key=='algorithm':
            if param_list[key] not in ['auto', 'brute', 'kd_tree', 'ball_tree']:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = param_list[key]
        else:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])

    if predictor_type=='regression':
        predictor = KNeighborsRegressor(n_neighbors=param_value_re['n_neighbors'], weights=param_value_re['weights'], p=param_value_re['p'], algorithm=param_value_re['algorithm'])
    else:
        predictor = KNeighborsClassifier(n_neighbors=param_value_re['n_neighbors'], weights=param_value_re['weights'], p=param_value_re['p'], algorithm=param_value_re['algorithm'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor
