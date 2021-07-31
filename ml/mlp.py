#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
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
    'hidden_layer_sizes':[100],
    'activation':'relu',
    'solver':'adam',
    'alpha':0.0001,
    'batch_size':'auto',
    'learning_rate':'constant',
    'learning_rate_init':0.001,
    'power_t':0.5,
    'max_iter':200,
    'suffle':True,
    'random_state':None,
    'tol':1e-4,
    'verbose':False,
    'warm_start':False,
    'momentum':0.9,
    'nesterovs_momentum':True,
    'early_stopping':False,
    'validation_fraction':0.1,
    'beta_1':0.9,
    'beta_2':0.999,
    'epsilon':1e-8,
    'n_iter_no_change':10,
    'max_fun':15000
    }

    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='hidden_layer_sizes':
            flag = True
            if param_list[key][0]!='[' or param_list[key][1]!=']':
                flag = False
            else:
                temp_len = len(param_list[key])
                size_list = param_list[key][1:temp_len-1].split(',')
                temp_list = []
                for size in size_list:
                    if process_str.check_letter(size)==True:
                        flag = False
                        break
                    else:
                        temp_list.append(int(size))
            if flag==True:
                param_value_re[key] = temp_list
            else:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
        elif key=='activation':
            if param_list[key] not in ['identity', 'logistic', 'tanh', 'relu']:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = param_list[key]

    if predictor_type=='regression':
        predictor = MLPRegressor(hidden_layer_sizes=param_value_re['hidden_layer_sizes'], activation=param_value_re['activation'])
    else:
        predictor = MLPClassifier(hidden_layer_sizes=param_value_re['hidden_layer_sizes'], activation=param_value_re['activation'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor
