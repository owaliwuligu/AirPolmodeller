#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
import xgboost as xgb
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from script import validation
from script import process_str

def train(X_train, Y_train, predictor_type, param_list, OneVsRest = False):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #define the params
    param_value_re = {
    'loss':'ls',
    'learning_rate':0.1,
    'n_estimators':100,
    'subsample':1.0,
    'criterion':'friedman_mse',
    'min_samples_split':2,
    'min_samples_leaf':1,
    'min_weight_fraction_leaf':0.,
    'max_depth':3,
    'min_impurity_decrease':0.,
    'init':None,
    'random_state':None,
    'max_features':None,
    'alpha':0.9,
    'verbose':0,
    'max_leaf_nodes':None,
    'warm_start':False,
    'validation_fraction':0.1,
    'n_iter_no_change':None,
    'tol':1e-4
    }
    if predictor_type=='classification':
        param_value_re['loss'] = 'deviance'
    #update the params with parma_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='loss':
            if predictor_type=='regression':
                if param_list[key] not in ['ls', 'lad', 'huber', 'quantile']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of teh parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]
            else:
                #loss:expotional just use for binary classification
                if param_list[key] not in ['deviance']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of teh parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]

        elif key in ['learning_rate', 'subsample', 'min_weight_fraction_leaf', 'min_impurity_decrease', 'alpha', 'validation_fraction', 'tol', 'ccp_alpha']:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            elif key=='ccp_alpha' and param_list[key]<0:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = float(param_list[key])
        elif key in ['n_estimators', 'max_depth', 'random_state', 'verbose']:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        elif key in ['subsample', 'min_weight_fraction_leaf', 'min_impurity_decrease', 'alpha', 'validation_fraction']:
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = float(param_list[key])
        elif key=='criterion':
            if param_list[key] not in ['friedman_mse', 'mae', 'mse']:
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
        elif key=='init':
            if param_list[key]=='zero':
                param_value_re[key] = 'zero'
            elif param_list[key]=='None':
                param_value_re[key] = None
        elif key=='max_features':
            if str(param_list[key])=='None':
                param_value_re[key] = None
            elif str(param_list[key]) in ['auto', 'sqrt', 'log2']:
                param_value_re[key] = param_list[key]
            elif re.search('\.', str(param_list[key]))!=None:
                param_value_re[key] = float(param_list[key])
            else:
                param_value_re[key] = int(param_list[key])
        elif key in ['n_iter_no_change', 'max_leaf_nodes']:
            if str(param_list[key])=='None':
                param_value_re[key] = None
            elif process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
    if predictor_type=='regression':
        predictor = GradientBoostingRegressor(loss=param_value_re['loss'], learning_rate=param_value_re['learning_rate'], n_estimators=param_value_re['n_estimators'], subsample=param_value_re['subsample'], criterion=param_value_re['criterion'], max_depth=param_value_re['max_depth'], min_samples_split=param_value_re['min_samples_split'], min_samples_leaf=param_value_re['min_samples_leaf'], min_weight_fraction_leaf=param_value_re['min_weight_fraction_leaf'], init=param_value_re['init'], max_features=param_value_re['max_features'], random_state=param_value_re['random_state'], alpha=param_value_re['alpha'], verbose=param_value_re['verbose'], max_leaf_nodes=param_value_re['max_leaf_nodes'], min_impurity_decrease=param_value_re['min_impurity_decrease'], warm_start=param_value_re['warm_start'], n_iter_no_change=param_value_re['n_iter_no_change'], tol=param_value_re['tol'])
    else:
        predictor = GradientBoostingClassifier(loss=param_value_re['loss'], learning_rate=param_value_re['learning_rate'], n_estimators=param_value_re['n_estimators'], subsample=param_value_re['subsample'], criterion=param_value_re['criterion'], max_depth=param_value_re['max_depth'], min_samples_split=param_value_re['min_samples_split'], min_samples_leaf=param_value_re['min_samples_leaf'], min_weight_fraction_leaf=param_value_re['min_weight_fraction_leaf'], init=param_value_re['init'], max_features=param_value_re['max_features'], random_state=param_value_re['random_state'], verbose=param_value_re['verbose'], max_leaf_nodes=param_value_re['max_leaf_nodes'], min_impurity_decrease=param_value_re['min_impurity_decrease'], warm_start=param_value_re['warm_start'], n_iter_no_change=param_value_re['n_iter_no_change'], tol=param_value_re['tol'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor

