#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
#import xgboost as xgb
import numpy as np
from ml import uni_method
from bayes_opt import BayesianOptimization
from script import read_code
from script import validation
from ml import xgb
from ml import linear
from ml import svm
from ml import knn
from ml import decision_tree
from ml import random_forest
from ml import gradient_boosting
from ml import extra_tree
from ml import adaboost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

#path=''
X=[]
Y=[]

def k_fold(X, Y, folds, predictor, validation_type):
    if folds=='leave-one-out':
        k = len(Y)
    else:
        k = int(folds)

    kf = KFold(k, True)
    kf.get_n_splits(X)

    labels = []
    for yy in Y:
        if yy not in labels:
            labels.append(yy)

    mae = 0.0 #mean absolute error
    evs = 0.0 #expained variance score
    mse = 0.0 #mean squared error
    r2s = 0.0 #R^2 score
    acc = 0.0 #macro-averaging
    rec = 0.0 #recall score
    f1s = 0.0 #f1 score
    mcc = 0.0 #mathews corrcoef
    fpr = 0.0
    tpr = 0.0
    pre = 0.0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if validation_type=='regression':
#            print("test")
            Y_predict = uni_method.predict(predictor, X_test)

            mae += mean_absolute_error(Y_test, Y_predict)
            evs += explained_variance_score(Y_test, Y_predict)
            mse += mean_squared_error(Y_test, Y_predict)
            r2s += r2_score(Y_test, Y_predict)
        if validation_type=='classification':
            Y_predict = uni_method.predict(predictor, X_test)

            Y_predict = label_binarize(Y_predict, labels)
            Y_test = label_binarize(Y_test, labels)
            #set types
            n_classes = Y_predict.shape[1]
            temp_recall = 0
            temp_acc_multi_binary = 0
            temp_fpr = 0.
            temp_mcc = 0.
            temp_tpr = 0.
            temp_precision = 0.
            for i in range(n_classes):
                temp_tp, temp_tn, temp_fp, temp_fn = 0 , 0 , 0 , 0
                # temp_recall += recall_score(y_test[:, i], y_pred_xgb[:, i])
                # temp_acc_multi_binary += accuracy_score(y_test[:, i], y_pred_xgb[:, i])
                for j in range(Y_test.shape[0]):
                    if Y_test[j][i]==1 and Y_predict[j][i]==1:
                        temp_tp+=1
                    elif Y_test[j][i]==0 and Y_predict[j][i]==0:
                        temp_tn+=1
                    elif Y_test[j][i]==1 and Y_predict[j][i]==0:
                        temp_fn+=1
                    elif Y_test[j][i]==0 and Y_predict[j][i]==1:
                        temp_fp+=1
                if temp_tp+temp_fn>0:
                    temp_recall += (temp_tp)/(temp_tp+temp_fn)
                if temp_tp+temp_tn+temp_fp+temp_fn>0:
                    temp_acc_multi_binary += (temp_tp+temp_tn)/(temp_tp+temp_tn+temp_fp+temp_fn)
                if temp_fp+temp_tn>0:
                    temp_fpr += (temp_fp)/(temp_fp+temp_tn)
                if (temp_tp+temp_fp)*(temp_tp+temp_fn)*(temp_tn+temp_fp)*(temp_tn+temp_fn)>0:
                    temp_mcc += (temp_tp*temp_tn-temp_fp*temp_fn)/np.sqrt((temp_tp+temp_fp)*(temp_tp+temp_fn)*(temp_tn+temp_fp)*(temp_tn+temp_fn))
                if temp_tp+temp_fn>0:
                    temp_tpr += (temp_tp)/(temp_tp+temp_fn)
                if temp_tp+temp_fp>0:
                    temp_precision += (temp_tp)/(temp_tp+temp_fp)
            rec += (temp_recall/n_classes)
            acc += (temp_acc_multi_binary/n_classes)
            fpr += (temp_fpr/n_classes)
            mcc += (temp_mcc/n_classes)
            tpr += (temp_tpr/n_classes)
            pre += (temp_precision/n_classes)          
    mae /= k
    evs /= k
    mse /= k
    r2s /= k

    pre /= k
    acc /= k
    rec /= k

    mcc /= k
    fpr /= k
    tpr /= k
    if pre+rec>0:
        f1s = (2*pre*rec)/(pre+rec)
#    thresholds /= k
    return mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre

def get_mse_xgb(max_depth, gamma, eta, learning_rate, n_estimators, min_child_weight, max_delta_step, subsample, colsample_bytree):
    param_value = {
    'booster':'gbtree',
    'max_depth':int(max_depth),
    'gamma':gamma,
    'eta':eta,
    'objective':'reg:gamma',
    'silent':False,
    'learning_rate':learning_rate,
    'n_estimators':int(n_estimators),
    'min_child_weight':min_child_weight,
    'max_delta_step':int(max_delta_step),
    'subsample':max(min(subsample, 1), 0),
    'colsample_bytree':max(min(subsample, 1), 0)
    }
    global X
    global Y
    #X,Y,headlines = read_code.read_training_file(path)
#    X,Y,headlines = read_code.read_training_file('/ipathogen-data/dongxu/AirPollution/example/air_data.txt')
    predictor = xgb.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
        
    return -mse

def get_mse_linear(fit_intercept, normalize, copy_X):
    param_value = {
    'fit_intercept':int(fit_intercept),
    'normalize':int(normalize),
    'copy_X':int(copy_X),
    'n_jobs':1
    }
    global X
    global Y
    #X,Y,headlines = read_code.read_training_file(path)
#    X,Y,headlines = read_code.read_training_file('/ipathogen-data/dongxu/AirPollution/example/air_data.txt')
    predictor = linear.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')

    return -mse

def knn_weights_select(select_id):
    if select_id==0:
        return 'uniform'
    else:
        return 'distance'

def get_mse_knn(n_neighbors, weights, p):
    weights_value = knn_weights_select(int(weights))
    param_value = {
    'n_neighbors':int(n_neighbors),
    'weights':weights_value,
    'p':int(p),
    'algorithm':'auto'
    }
    predictor = knn.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse

def svm_kernel_select(select_id):
    if select_id==0:
        return 'linear'
    elif select_id==1:
        return 'rbf'
    elif select_id==2:
        return 'poly'
    elif select_id==3:
        return 'sigmoid'
    elif select_id==4:
        return 'precomputed'

def get_mse_svm(kernel, degree, gamma, coef0, tol, C, shrinking, epsilon):
    kernel_value = svm_kernel_select(int(kernel))
    param_value = {
    'kernel':'rbf',
    'degree':int(degree),
    'gamma':gamma,
    'coef0':coef0,
    'tol':tol,
    'C':C,
    'shrinking':int(shrinking),
    'epsilon':epsilon
    }
    global X
    global Y
    predictor = svm.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')

    return -mse

def dt_criterion_select_re(select_id):
    if select_id==0:
        return 'mse'
    elif select_id==1:
        return 'friedman_mse'
    else:
        return 'mae'

def dt_splitter_select(select_id):
    if select_id==0:
        return 'best'
    else:
        return 'random'

def get_mse_dt(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease):
    criterion_value = dt_criterion_select_re(int(criterion))
    splitter_value = dt_splitter_select(int(splitter))
    param_value = {
    'criterion':criterion_value,
    'splitter':splitter_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = decision_tree.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse

def ad_loss_select(select_id):
    if select_id==0:
        return 'linear'
    elif select_id==1:
        return 'square'
    else:
        return 'exponential'

def get_mse_ad(n_estimators, learning_rate, loss, random_state):
    loss_value = ad_loss_select(int(loss))
    param_value={
    'n_estimators':int(n_estimators),
    'learning_rate':learning_rate,
    'loss':loss_value,
    'random_state':int(random_state)
    }
    global X
    global Y
    predictor = adaboost.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse

def get_mse_et(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease):
    criterion_value = dt_criterion_select_re(int(criterion))
    splitter_value = dt_splitter_select(int(splitter))
    param_value = {
    'criterion':criterion_value,
    'splitter':splitter_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = extra_tree.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse


def dt_criterion_select_clf(select_id):
    if select_id==0:
        return 'gini'
    else:
        return 'entropy'

def get_mse_rf(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, random_state):
    criterion_value = dt_criterion_select_re(int(criterion))
    param_value = {
    'criterion':criterion_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = random_forest.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse

def gbdt_loss_select_re(select_id):
    if select_id==0:
        return 'ls'
    elif select_id==1:
        return 'lad'
    elif select_id==2:
        return 'huber'
    else:
        return 'quantile'
def gbdt_loss_select_clf(select_id):
    if select_id==0:
        return 'deviance'
#    else:
#        return 'exponential'

def get_mse_gbdt(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, random_state, max_features, max_leaf_nodes):
    criterion_value = dt_criterion_select_re(int(criterion))
    loss_value = gbdt_loss_select_re(int(loss))
    param_value = {
    'loss':loss_value,
    'learning_rate':learning_rate,
    'n_estimators':int(n_estimators),
    'subsample':subsample,
    'criterion':criterion_value,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_depth':int(max_depth),
    'min_impurity_decrease':min_impurity_decrease,
    'random_state':int(random_state),
    'max_features':max_features,
    'max_leaf_nodes':int(max_leaf_nodes)
    }
    global X
    global Y
    predictor = gradient_boosting.train(X, Y, 'regression', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'regression')
    return -mse

def get_acc_gbdt(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, random_state, max_features, max_leaf_nodes):
    criterion_value = dt_criterion_select_re(int(criterion))
    loss_value = gbdt_loss_select_clf(int(loss))
    param_value = {
    'loss':loss_value,
    'learning_rate':learning_rate,
    'n_estimators':int(n_estimators),
    'subsample':subsample,
    'criterion':criterion_value,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_depth':int(max_depth),
    'min_impurity_decrease':min_impurity_decrease,
    'random_state':int(random_state),
    'max_features':max_features,
    'max_leaf_nodes':int(max_leaf_nodes)
    }
    global X
    global Y
    predictor = gradient_boosting.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc


def get_acc_dt(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease):
    criterion_value = dt_criterion_select_clf(int(criterion))
    splitter_value = dt_splitter_select(int(splitter))
    param_value = {
    'criterion':criterion_value,
    'splitter':splitter_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = decision_tree.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc

def ad_algorithm_select(select_id):
    if select_id==0:
        return 'SAMME'
    else:
        return 'SAMME.R'

def get_acc_ad(n_estimators, learning_rate, algorithm, random_state):
    algorithm_value = ad_algorithm_select(int(algorithm))
    param_value={
    'n_estimators':int(n_estimators),
    'learning_rate':learning_rate,
    'algorithm':algorithm_value,
    'random_state':int(random_state)
    }
    global X
    global Y
    predictor = adaboost.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc


def get_acc_et(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease):
    criterion_value = dt_criterion_select_clf(int(criterion))
    splitter_value = dt_splitter_select(int(splitter))
    param_value = {
    'criterion':criterion_value,
    'splitter':splitter_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = extra_tree.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc


def get_acc_rf(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, random_state):
    criterion_value = dt_criterion_select_clf(int(criterion))
    param_value = {
    'criterion':criterion_value,
    'max_depth':int(max_depth),
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'min_weight_fraction_leaf':min_weight_fraction_leaf,
    'max_features':max_features,
    'random_state':int(random_state),
    'max_leaf_nodes':int(max_leaf_nodes),
    'min_impurity_decrease':min_impurity_decrease
    }
    global X
    global Y
    predictor = random_forest.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc

def get_acc_knn(n_neighbors, weights, p):
    weights_value = knn_weights_select(int(weights))
    param_value = {
    'n_neighbors':int(n_neighbors),
    'weights':weights_value,
    'p':int(p),
    'algorithm':'auto'
    }
    predictor = knn.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')
    return acc


def get_acc_xgb(max_depth, gamma, eta, learning_rate, n_estimators, min_child_weight, max_delta_step, subsample, colsample_bytree):
    param_value = {
    'booster':'gbtree',
    'max_depth':int(max_depth),
    'gamma':gamma,
    'eta':eta,
    'objective':'reg:gamma',
    'silent':False,
    'learning_rate':learning_rate,
    'n_estimators':int(n_estimators),
    'min_child_weight':min_child_weight,
    'max_delta_step':int(max_delta_step),
    'subsample':max(min(subsample, 1), 0),
    'colsample_bytree':max(min(subsample, 1), 0)
    }
    global X
    global Y
    #X,Y,headlines = read_code.read_training_file(path)
#    X,Y,headlines = read_code.read_training_file('/ipathogen-data/dongxu/AirPollution/example/air_data.txt')
    predictor = xgb.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')

    return acc

def get_acc_svm(kernel, degree, gamma, coef0, tol, C, shrinking):
    kernel_value = svm_kernel_select(int(kernel))
    param_value = {
    'kernel':'rbf',
    'degree':int(degree),
    'gamma':gamma,
    'coef0':coef0,
    'tol':tol,
    'C':C,
    'shrinking':int(shrinking)
    }
    global X
    global Y
    predictor = svm.train(X, Y, 'classification', param_value)
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre = k_fold(np.array(X), np.array(Y), 5, predictor, 'classification')

    return acc

def optimize(method, predictor_type, X_train, Y_train):
    #rf_bo = BayesianOptimization(get_evs_xgb, {'booster':'gbtree', 'max_depth':(3, 15), 'gamma':(0, 10.0), 'eta':(0.01, 0.3), 'objective':'reg:gamma', 'silent':False, 'learning_rate':(0.01, 0.5), 'n_estimators':(10, 250), 'min_child_weight':(0, 20), 'max_delta_step':(0, 10), 'subsample':(0.4, 1.0), 'colsample_bytree':(0.4, 1.0), 'colsample_bylevel':1})
    global X
    global Y
    X = X_train
    Y = Y_train
    print(method)
    if method=='xgboost':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_xgb, {'max_depth':(3, 15), 'gamma':(0, 0.99), 'eta':(0.01, 0.3), 'learning_rate':(0.01, 0.5), 'n_estimators':(10, 250), 'min_child_weight':(0.001, 20), 'max_delta_step':(1, 10), 'subsample':(0.4, 1.0), 'colsample_bytree':(0.4, 1.0)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
        else:
            rf_bo = BayesianOptimization(get_acc_xgb, {'max_depth':(3, 15), 'gamma':(0, 0.99), 'eta':(0.01, 0.3), 'learning_rate':(0.01, 0.5), 'n_estimators':(10, 250), 'min_child_weight':(0.001, 20), 'max_delta_step':(1, 10), 'subsample':(0.4, 1.0), 'colsample_bytree':(0.4, 1.0)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
        bo_res = rf_bo.max['params']
    elif method=='linear':
        rf_bo = BayesianOptimization(get_mse_linear, {'fit_intercept':(0.001, 1.99), 'normalize':(0.001, 1.99), 'copy_X':(0.001, 1.99)}, random_state=1234, verbose=2)
        rf_bo.maximize(n_iter=15)
        print(rf_bo.max)
        bo_res = rf_bo.max['params']
    elif method=='svm':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_svm, {'kernel':(1, 4.99), 'degree':(2, 6), 'gamma':(1.0/len(Y), 0.99), 'coef0':(0.001, 1.0), 'tol':(0.01, 0.1), 'C':(0.001, 1.0), 'shrinking':(0.001, 1.99), 'epsilon':(0.001, 1.0)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
        else:
            rf_bo = BayesianOptimization(get_acc_svm, {'kernel':(1, 4.99), 'degree':(2, 6), 'gamma':(1.0/len(Y), 0.99), 'coef0':(0.001, 1.0), 'tol':(0.01, 0.1), 'C':(0.001, 1.0), 'shrinking':(0.001, 1.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
        bo_res = rf_bo.max['params']
        bo_res['kernel'] = 'rbf'
    elif method=='knn':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_knn, {'n_neighbors':(5,20.99), 'weights':(0.001, 1.99), 'p':(1,6.99)})
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
        else:
            rf_bo = BayesianOptimization(get_acc_knn, {'n_neighbors':(5,20.99), 'weights':(0.001, 1.99), 'p':(1,6.99)})
            rf_bo.maximize(n_iter=15)
            print(rf_bo.max)
#        rf_bo.max['params']['weights']=knn_weights_select(int(rf_bo.max['params']['weights']))
        bo_res = rf_bo.max['params']
        bo_res['weights'] = knn_weights_select(int(bo_res['weights']))
    elif method=='decision_tree':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_dt, {'criterion':(0,2.99), 'splitter':(0,1.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001,0.5), 'max_features':(0.001,1), 'random_state':(0,100.99), 'max_leaf_nodes':(3,50.99), 'min_impurity_decrease':(0.001,1)})
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_re(int(bo_res['criterion']))
        else:
            rf_bo = BayesianOptimization(get_acc_dt, {'criterion':(0,2.99), 'splitter':(0,1.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001,0.5), 'max_features':(0.001,1), 'random_state':(0,100.99), 'max_leaf_nodes':(3,50.99), 'min_impurity_decrease':(0.001,1)})
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_clf(int(bo_res['criterion']))
        bo_res['splitter'] = dt_splitter_select(int(bo_res['splitter']))
    elif method=='extra_tree':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_et, {'criterion':(0,2.99), 'splitter':(0,1.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001,0.5), 'max_features':(0.001,1), 'random_state':(0,100.99), 'max_leaf_nodes':(3,50.99), 'min_impurity_decrease':(0.001,1)})
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_re(int(bo_res['criterion']))
        else:
            rf_bo = BayesianOptimization(get_acc_et, {'criterion':(0,2.99), 'splitter':(0,1.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001,0.5), 'max_features':(0.001,1), 'random_state':(0,100.99), 'max_leaf_nodes':(3,50.99), 'min_impurity_decrease':(0.001,1)})
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_clf(int(bo_res['criterion']))
        bo_res['splitter'] = dt_splitter_select(int(bo_res['splitter']))
    elif method=='random_forest':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_rf, {'n_estimators':(10, 250), 'criterion':(0, 2.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001, 0.5), 'max_features':(0.001, 1), 'max_leaf_nodes':(3, 50.99), 'min_impurity_decrease':(0.001, 1), 'random_state':(0, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_re(int(bo_res['criterion']))
        else:
            rf_bo = BayesianOptimization(get_acc_rf, {'n_estimators':(10, 250), 'criterion':(0, 2.99), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001, 0.5), 'max_features':(0.001, 1), 'max_leaf_nodes':(3, 50.99), 'min_impurity_decrease':(0.001, 1), 'random_state':(0, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_clf(int(bo_res['criterion']))
    elif method=='gradient_boosting':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_gbdt, {'loss':(0, 3.99), 'learning_rate':(0.01, 0.1), 'n_estimators':(10, 250), 'criterion':(0, 2.99), 'subsample':(0.8, 1.0), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001, 0.5), 'max_features':(0.001, 1), 'max_leaf_nodes':(3, 50.99), 'min_impurity_decrease':(0.001, 1), 'random_state':(0, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_re(int(bo_res['criterion']))
            bo_res['loss'] = gbdt_loss_select_re(int(bo_res['loss']))
        else:
            rf_bo = BayesianOptimization(get_acc_gbdt, {'loss':(0, 0.99), 'learning_rate':(0.01, 0.1), 'n_estimators':(10, 250), 'criterion':(0, 2.99), 'subsample':(0.8, 1.0), 'max_depth':(3, 15.99), 'min_samples_split':(0.001, 1), 'min_samples_leaf':(0.001, 0.5), 'min_weight_fraction_leaf':(0.001, 0.5), 'max_features':(0.001, 1), 'max_leaf_nodes':(3, 50.99), 'min_impurity_decrease':(0.001, 1), 'random_state':(0, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['criterion'] = dt_criterion_select_re(int(bo_res['criterion']))
            bo_res['loss'] = gbdt_loss_select_clf(int(bo_res['loss']))
    elif method=='adaboost':
        if predictor_type=='regression':
            rf_bo = BayesianOptimization(get_mse_ad, {'n_estimators':(20,100.99), 'learning_rate':(0.1, 1.0), 'loss':(0, 2.99), 'random_state':(1, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['loss'] = ad_loss_select(int(bo_res['loss']))
        else:
            rf_bo = BayesianOptimization(get_acc_ad, {'n_estimators':(20,100.99), 'learning_rate':(0.1, 1.0), 'algorithm':(0, 1.99), 'random_state':(1, 100.99)}, random_state=1234, verbose=2)
            rf_bo.maximize(n_iter=15)
            bo_res = rf_bo.max['params']
            bo_res['algorithm'] = ad_algorithm_select(int(bo_res['algorithm']))
    return bo_res
#optimize()   
