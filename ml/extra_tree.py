#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
import xgboost as xgb
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from script import validation
from script import process_str

def train(X_train, Y_train, predictor_type, param_list, OneVsRest=False):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define the parmas
    param_value_re = {
    'criterion':'mse',
    'splitter':'random',
    'max_depth':None,
    'min_samples_split':2,
    'min_samples_leaf':1,
    'min_weight_fraction_leaf':0.0,
    'max_features':'auto',
    'random_state':None,
    'min_impurity_decrease':0.0,
    'max_leaf_nodes':None
    }
    
    if predictor_type=='classification':
        param_value_re['criterion'] = 'gini'
    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='criterion':
            if predictor_type=='regression':
                if param_list[key] not in ['mse', 'friedman_mse', 'mae']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]
            else:
                if param_list[key] not in ['gini', 'entropy']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of teh parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]
        elif key=='splitter':
            if param_list[key] not in ['best', 'random']:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)

            else:
                param_value_re[key] = param_list[key]
        elif key in ['min_samples_split', 'min_samples_leaf']:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            elif re.search('\.', str(param_list[key]))!=None:
                param_value_re[key] = float(param_list[key])
            else:
                param_vluae_re[key] = int(param_list[key])
        elif key in ['max_depth', 'random_state', 'max_leaf_nodes']:
            if str(param_list[key])=='None':
                param_value_re[key] = None
            elif process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        elif key in ['min_weight_fraction_leaf', 'min_impurity_decrease']:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = float(param_list[key])
        elif key=='max_features':
            if str(param_list[key])=='None':
                param_value_re[key] = None
            elif str(param_list[key]) in ['auto', 'sqrt', 'log2']:
                param_value_re[key] = param_list[key]
            elif re.search('\.', str(param_list[key]))!=None:
                param_value_re[key] = float(param_list[key])
            else:
                param_value_re[key] = int(param_list[key])
    if predictor_type=='regression':
        predictor = ExtraTreeRegressor(criterion=param_value_re['criterion'], splitter=param_value_re['splitter'], max_depth=param_value_re['max_depth'], min_samples_split=param_value_re['min_samples_split'], min_samples_leaf=param_value_re['min_samples_leaf'], min_weight_fraction_leaf=param_value_re['min_weight_fraction_leaf'], max_features=param_value_re['max_features'], random_state=param_value_re['random_state'], max_leaf_nodes=param_value_re['max_leaf_nodes'], min_impurity_decrease=param_value_re['min_impurity_decrease'])
    else:
        predictor = ExtraTreeClassifier(criterion=param_value_re['criterion'], splitter=param_value_re['splitter'], max_depth=param_value_re['max_depth'], min_samples_split=param_value_re['min_samples_split'], min_samples_leaf=param_value_re['min_samples_leaf'], min_weight_fraction_leaf=param_value_re['min_weight_fraction_leaf'], max_features=param_value_re['max_features'], random_state=param_value_re['random_state'], max_leaf_nodes=param_value_re['max_leaf_nodes'], min_impurity_decrease=param_value_re['min_impurity_decrease'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor
        
