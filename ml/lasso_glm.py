#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn import linear_model
from script import process_str
import numpy as np

def train(X_train, Y_train, predictor_type, param_list):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    if predictor_type=='classification':
        print('The linear model can just be used with regression!\n')
        sys.stderr.write('>The linear model can just be used with regression!\n')
        sys.exit(1)
    #define the params
    param_value_re = {
    'alpha':0.1
    }
    #update the params with param_list
    for key in param_list.keys():
        if key not in param_value_re.keys():
            print('The parameter \''+ key + '\' is not valid, so it will be skipped!\n')
        else:
            if key in ['alpha']:
                if process_str.check_letter(str(param_list[key]))==True:
                    print('The value of the parameter \''+key+'\' is not valid!\n')
                    sys.stderr.write('>The value of the parameter \''+key+'\' is not valid!\n')
                    sys.exit(1)
                else:
                    param_value_re[key] = float(param_list[key])
    #just regression
    predictor = linear_model.Lasso(alpha=param_value_re['alpha'])
    predictor.fit(X_train, Y_train)

    return predictor
