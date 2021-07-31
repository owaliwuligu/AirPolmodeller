import sys,re,os
import xgboost as xgb
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multiclass import OneVsRestClassifier
from script import validation
from script import process_str
from ml import uni_method

def get_params_from_str(params):
    method = params.split('@')[0].strip(' ')
    if len(params.split('@'))>1:
        params_str = params.split('@')[1].strip(' ')
    else:
        params_str=''
        method=params.split('#')[0].strip(' ')
    if params_str=='':
        return method, None
    param_list = params_str.split('#')
    param_res = ''
    
    for param_id in range(0, len(param_list)-1):
        param_res += param_list[param_id].split('=')[0].strip(' ')
        param_res += (':'+param_list[param_id].split('=')[1].strip(' ')+',')
    param_res += param_list[len(param_list)-1].split('=')[0].strip(' ')
    param_res += (':'+param_list[len(param_list)-1].split('=')[1].strip(' '))
    return method, param_res

def train(X_train, Y_train, predictor_type, param_list, OneVsRest=False):

    #define the params
    param_value_re = {
    'base_estimator':None,    
    'n_estimators':50,
    'learning_rate':1.0,
    'loss':'linear',
    'random_state':None
    }

    if predictor_type=='classification':
        param_value_re['algorithm']='SAMME.R'

    base_estimator_value = None
    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
            continue
        if key=='base_estimator':
            method, base_param = get_params_from_str(param_list[key])
            if method in ['bagging', 'adaboost', 'stacking']:
                print('Please choose a valid base estimator!\n')
                sys.stderr.write('>Please choose a valid base estimator!\n')
                sys.exit(1)
            else:
                base_estimator_value, none_temp = uni_method.train(method, X_train, Y_train, predictor_type, base_param)
        elif key=='loss':
            if predictor_type=='regression':
                if param_list[key] not in ['linear', 'square', 'exponential']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]
            else:
                print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
                continue
        elif key=='n_estimators':
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        elif key=='learning_rate':
            if process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = float(param_list[key])
        elif key=='random_state':
            if str(param_list[key])=='None':
                param_value_re[key] = None
            elif process_str.check_letter(str(param_list[key]))==True:
                print('The value of the parameter \''+key+'\' is not valid!\n')
                sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                sys.exit(1)
            else:
                param_value_re[key] = int(param_list[key])
        elif key=='algorithm':
            if predictor_type=='regression':
                print('The parameter \''+key+'\' is not valid, so it will be skipped!\n')
                continue
            else:
                if param_list[key] not in ['SAMME', 'SAMME.R']:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = param_list[key]
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    if predictor_type=='regression':
        predictor = AdaBoostRegressor(base_estimator=base_estimator_value, n_estimators=param_value_re['n_estimators'], learning_rate=param_value_re['learning_rate'], loss=param_value_re['loss'], random_state=param_value_re['random_state'])
    else:
        predictor = AdaBoostClassifier(base_estimator=base_estimator_value, n_estimators=param_value_re['n_estimators'], learning_rate=param_value_re['learning_rate'], algorithm=param_value_re['algorithm'], random_state=param_value_re['random_state'])
    if OneVsRest==True:
        predictor = OneVsRestClassifier(predictor)
    predictor.fit(X_train, Y_train)
    return predictor

