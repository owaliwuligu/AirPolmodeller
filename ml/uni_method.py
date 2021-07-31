#!/usr/bin/env python
#_*_coding:utf-8_*_
import pickle
import re,os,sys
import tarfile
import shutil
from script import process_str
from ml import xgb
from ml import svm
from ml import linear
from ml import bagging
from ml import knn
from ml import decision_tree
from ml import random_forest
from ml import gradient_boosting
from ml import extra_tree
from ml import adaboost
from ml import stacking
from ml import deep_ensemble
from ml import lasso_glm
from ml import mlp
#from ml import deep_ensemble
import numpy as np
from ml import bayesian_optimization

def train(ml_method, X, Y, predictor_type='regression', params=None, param_value_bo=None, OneVsRest=False):
#    predictor_type='classification'

    param_value = {}
    method_param_value = {}
    method_param_value2 = {}

#    for y in Y:
#        if re.search('\.', str(y))!=None:
#            predictor_type='regression'
#            break
    for y in Y:
        if re.search('\.', str(y))!=None and predictor_type=='classification':
            print('Classification can not be used for numeric data!\n')
            sys.stderr.write('>Classification can not be used for numeric data!\n')
            sys.exit(1);

    if params!=None:
        if params=='BO':
            #param_value_bo is None means the programm is at the bayeisian optimization state; when param_value_bo is not None, it meas the programm is at a validation stage
            if param_value_bo==None:
                bo_res = bayesian_optimization.optimize(ml_method, predictor_type, X, Y)
                param_value = bo_res
            else:
                param_value = param_value_bo
#            for key in best_params.keys():
#                param_value[key] = best_params[key] 
        elif ml_method not in ['bagging', 'stacking', 'deep_ensemble']:
            param_list = params.split(',')
            for p in param_list:
                temp_list = p.split(':')
#                flag = False
#                for key in param_value.keys():
#                    if key==temp_list[0]:
#                        flag = True
                temp_value = temp_list[1].strip('\n')
                key = temp_list[0]
                temp_value = temp_value.strip('\r')
#                print(temp_value)
                if process_str.check_letter(temp_value)==True:
#                            print(temp_value)
                    if temp_value=='True':
                        param_value[key] = True
                    elif temp_value=='False':
                        param_value[key] = False
                    else:
                        param_value[key] = temp_value
                else:
                    if re.search('\.', temp_value)!=None:
                        param_value[key] = float(temp_value)

                    else:
                        param_value[key] = int(temp_value)
        elif ml_method=='deep_ensemble':
            if re.search('BO', params)!=None and param_value_bo!=None:
                method_param_value = param_value_bo
            else:
                method_params_list = params.split('I')[0].split('#')
                method_params_list2 = params.split('I')[1].split('#')
                for j in method_params_list:
                    print(j)
                print('-------------------')
                for i in method_params_list2:
                    print(i)
                mp_index = 0
                mp_index2 = 0
                for mp in method_params_list:
                    method_temp = mp.split('@')[0]
                    param_list_temp = {}
                    if len(mp.split('@'))==1:
                        method_temp = method_temp + '-' + str(mp_index)
                        mp_index += 1
                        method_param_value[method_temp] = param_list_temp
                        continue
                    if mp.split('@')[1]=='BO':
                        method_temp0 = method_temp
                        bo_res = bayesian_optimization.optimize(method_temp0, predictor_type, X, Y)
                        param_list_temp = bo_res
                    else:
                        params_temp = mp.split('@')[1].split(',')
                        for pt in params_temp:
                            pt_list = pt.split(':')
                            key_temp = pt_list[0]
                            value_temp = pt_list[1].strip('\n')
                            value_temp = value_temp.strip('\r')
                            if process_str.check_letter(value_temp)==True:
                                if value_temp=='True':
                                    param_list_temp[key_temp] = True
                                elif value_temp=='False':
                                    param_list_temp[key_temp] = False
                                else:
                                    param_list_temp[key_temp] = value_temp
                            else:
                                if re.search('\.', value_temp)!=None:
                                    param_list_temp[key_temp] = float(value_temp)
                                else:
                                    param_list_temp[key_temp] = int(value_temp)
                    method_temp = method_temp + '-' + str(mp_index)
                    mp_index += 1
                    method_param_value[method_temp] = param_list_temp
                for mp in method_params_list2:
                    method_temp = mp.split('@')[0]
                    param_list_temp = {}
                    if len(mp.split('@'))==1:
                        method_temp = method_temp + '-' + str(mp_index2)
                        mp_index += 1
                        method_param_value2[method_temp] = param_list_temp
                        continue
                    if mp.split('@')[1]=='BO':
                        method_temp0 = method_temp
                        bo_res = bayesian_optimization.optimize(method_temp0, predictor_type, X, Y)
                        param_list_temp = bo_res
                    else:
                        params_temp = mp.split('@')[1].split(',')
                        for pt in params_temp:
                            pt_list = pt.split(':')
                            key_temp = pt_list[0]
                            value_temp = pt_list[1].strip('\n')
                            value_temp = value_temp.strip('\r')
                            if process_str.check_letter(value_temp)==True:
                                if value_temp=='True':
                                    param_list_temp[key_temp] = True
                                elif value_temp=='False':
                                    param_list_temp[key_temp] = False
                                else:
                                    param_list_temp[key_temp] = value_temp
                            else:
                                if re.search('\.', value_temp)!=None:
                                    param_list_temp[key_temp] = float(value_temp)
                                else:
                                    param_list_temp[key_temp] = int(value_temp)
                    method_temp = method_temp + '-' + str(mp_index2)
                    mp_index2 += 1
                    method_param_value2[method_temp] = param_list_temp
        else:
            if re.search('BO', params)!=None and param_value_bo!=None:
                method_param_value = param_value_bo
            else:
                method_params_list = params.split('#')
                mp_index = 0
                for mp in method_params_list:
                    method_temp = mp.split('@')[0]
                    param_list_temp = {}
                    if len(mp.split('@'))==1:
                        method_temp = method_temp + '-' + str(mp_index)
                        mp_index += 1
                        method_param_value[method_temp] = param_list_temp
                        continue
                    if mp.split('@')[1]=='BO':
                        method_temp0 = method_temp
                        if method_temp[len(method_temp)-1]=='*':
                            method_temp0 = method_temp[:len(method_temp)-1]
                        bo_res = bayesian_optimization.optimize(method_temp0, predictor_type, X, Y)
                        param_list_temp = bo_res
                    else:
                        params_temp = mp.split('@')[1].split(',')
                        for pt in params_temp:
                            pt_list = pt.split(':')
                            key_temp = pt_list[0]
                            value_temp = pt_list[1].strip('\n')
                            value_temp = value_temp.strip('\r')
                            if process_str.check_letter(value_temp)==True:
                                if value_temp=='True':
                                    param_list_temp[key_temp] = True
                                elif value_temp=='False':
                                    param_list_temp[key_temp] = False
                                else:
                                    param_list_temp[key_temp] = value_temp
                            else:
                                if re.search('\.', value_temp)!=None:
                                    param_list_temp[key_temp] = float(value_temp)
                                else:
                                    param_list_temp[key_temp] = int(value_temp)
                    method_temp = method_temp + '-' + str(mp_index)
                    mp_index += 1
                    method_param_value[method_temp] = param_list_temp
#            print(len(method_param_value.keys()))    
#                if flag==False:
#                    print('Warning: The parameter \''+temp_list[0]+'\' is not valid, so it will be skipped!\n')
#    for y in Y:
#        if re.search('\.', str(y))!=None:
#            predictor_type='regression'
#            break
    if ml_method=='xgboost':
        predictor = xgb.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='svm':
        predictor = svm.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='linear':
        if predictor_type=='classification':
            print('The linear model can just be used with regression!\n')
            sys.stderr.write('>The linear model can just be used with regression!\n')
            sys.exit(1)
        else:
            predictor = linear.train(X, Y,predictor_type, param_value)
    elif ml_method=='bagging':
        predictor = bagging.train(X, Y, predictor_type, method_param_value)
    elif ml_method=='knn':
        predictor = knn.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='decision_tree':
        predictor = decision_tree.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='random_forest':
        predictor = random_forest.train(X, Y, predictor_type, param_value, OneVsRest)   
    elif ml_method=='gradient_boosting':
        predictor = gradient_boosting.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='extra_tree':
        predictor = extra_tree.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='adaboost':
        predictor = adaboost.train(X, Y, predictor_type, param_value, OneVsRest)
    elif ml_method=='stacking':
        predictor = stacking.train(X, Y, predictor_type, method_param_value)
    elif ml_method=='deep_ensemble':
        predictor = deep_ensemble.train(X, Y, predictor_type, method_param_value, method_param_value2)
    elif ml_method=='lasso_glm':
        predictor = lasso_glm.train(X, Y, predictor_type, method_param_value)
    elif ml_method=='mlp':
        predictor = mlp.train(X, Y, predictor_type, method_param_value, OneVsRest)

    if params=='BO' and param_value_bo==None:
        return predictor, param_value
    elif params!=None and params!='BO' and re.search('BO', params)!=None and param_value_bo==None:
        return predictor, method_param_value
    else:
        return predictor, None

def predict(clf, X):
#    f = open(model_path, 'rb')
#    clf = pickle.load(f)
#    f.close()
#    X = np.array(X)
#    print(clf)
    Y = clf.predict(X)

    return Y

def load_model(model_path):
    if tarfile.is_tarfile(model_path):
#        print('bagging!')
        path_current = os.getcwd()
        model_parfolder_path = ''
        path_list = model_path.split('/')
        for index in range(len(path_list)-1):
            model_parfolder_path += path_list[index] + '/'
        os.chdir(model_parfolder_path)
        model_name = model_path.split('/')[len(model_path.split('/'))-1].strip('\r').strip('\n')
        with tarfile.open(model_name, 'r') as tar:
            tar.extractall()
        f_flat = open('models_dir_3A89BDdsiwa83Eaeek/flag')
        stratege = f_flat.readline().strip('\n')
        predictor_type = f_flat.readline().strip('\n')
        f_flat.close()
        if stratege=='bagging':
            model_list = []
            for f in os.listdir('models_dir_3A89BDdsiwa83Eaeek'):
                if f not in ['flag', 'dictionary', 'y_dictionary']:
                    f_model = open('models_dir_3A89BDdsiwa83Eaeek/'+f, 'rb')
                    predictor_temp = pickle.load(f_model)
                    f_model.close()
                    model_list.append(predictor_temp)
            clf = bagging.bagging_predictor(len(model_list), model_list, predictor_type)
        elif stratege=='deep_ensemble':
            model_A_list = []
            model_B_list = []
            weight_list = []
            for f in os.listdir('models_dir_3A89BDdsiwa83Eaeek/model'):
                if re.search('_A', f)!=None:
                    f_model = open('models_dir_3A89BDdsiwa83Eaeek/'+f, 'rb')
                    predictor_temp = pickle.load(f_model)
                    f_model.close()
                    model_A_list.append(predictor_temp)
                elif re.search('_B', f)!=None:
                    f_model = open('models_dir_3A89BDdsiwa83Eaeek/'+f, 'rb')
                    predictor_temp = pickle.load(f_model)
                    f_model.close()
                    model_B_list.append(predictor_temp)
                    weight_list.append(1)
                clf = deep_ensemble.deep_predictor(len(model_A_list), model_A_list, len(model_B_list), model_B_list, predictor_type, weight_list)
        elif stratege=='algorithm':
            f = open('models_dir_3A89BDdsiwa83Eaeek/model', 'rb')
            clf = pickle.load(f)
            f.close()
#        shutil.rmtree('models_dir_3A89BDdsiwa83Eaeek')
        os.chdir(path_current)
    else:
        print('The model is not correct!\n')
        sys.stderr.write('>The model is not correct!\n')
        sys.exit(1)
#    clf = pickle.load(open(model_path, 'rb'))
    return clf, model_parfolder_path
