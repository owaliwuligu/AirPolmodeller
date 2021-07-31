#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy as np
import pickle
from script import validation
from script import process_str
from sklearn.multiclass import OneVsRestClassifier
def train(X_train, Y_train, predictor_type, param_list, OneVsRest=False):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define the params  gamma:'auto' will be changed as gamma:'scale' in 0.22
    param_value_re = {
    'kernel':'rbf',
    'degree':3,
    'gamma':1.0/len(Y_train),
    'coef0':0.0,
    'tol':0.01,
    'C':1.0,
    'shrinking':True,
    'epsilon':0.1
    }
    
    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='epsilon' and predictor_type=='classification':
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='kernel':
            if param_list[key] not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = param_list[key]
        elif key=='gamma':
            if param_list[key] not in ['auto', 'scale', 'auto_deprecated']:
                if process_str.check_letter(str(param_list[key]))==True:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = float(param_list[key])
        elif key=='shrinking' or key=='degree':
            if param_list[key]!=True and param_list[key]!=False and process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        else:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = float(param_list[key])
    if predictor_type=='regression':
        predictor = SVR(kernel=param_value_re['kernel'], degree=param_value_re['degree'], gamma=param_value_re['gamma'], coef0=param_value_re['coef0'], tol=param_value_re['tol'], C=param_value_re['C'], shrinking=param_value_re['shrinking'], epsilon=param_value_re['epsilon'])
    else:
        predictor = SVC(kernel=param_value_re['kernel'], degree=param_value_re['degree'], gamma=param_value_re['gamma'], coef0=param_value_re['coef0'], tol=param_value_re['tol'], C=param_value_re['C'], shrinking=param_value_re['shrinking'], probability=True)
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor
