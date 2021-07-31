import sys,re,os
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from mlxtend.classifier import StackingClassifier
from mlxtend.regressor import StackingRegressor
from sklearn.multiclass import OneVsRestClassifier
from script import validation
from script import process_str
from ml import uni_method
from ml import xgb
from ml import linear
from ml import svm
from ml import decision_tree
from ml import knn
from ml import gradient_boosting
from ml import random_forest
from ml import extra_tree
from ml import lasso_glm
from ml import mlp
def train(X_train, Y_train, predictor_type, method_param_list):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    #define the params
    
    model_list = []
    meta_model = None
    print('test!')
    for mp_key in method_param_list.keys():
        real_mp_key = mp_key.split('-')[0]
        if_meta_model = False
        if re.search('\*', real_mp_key)!=None:
            rmk_len = len(real_mp_key)
            real_mp_key = real_mp_key[:rmk_len-1]
            if_meta_model = True
        print(real_mp_key)
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

        if if_meta_model==False:
            model_list.append(predictor)
        else:
            meta_model = predictor
    if predictor_type=='regression':
        stacking_predictor = StackingRegressor(regressors=model_list, meta_regressor=meta_model)
    else:
        stacking_predictor = StackingClassifier(classifiers=model_list, meta_classifier=meta_model)
#    if OneVsRest==True:
#        stacking_predictor = OneVsRestClassifier(stacking_predictor)
    stacking_predictor.fit(X_train, Y_train)
    return stacking_predictor
