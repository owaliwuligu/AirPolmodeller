#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
from sklearn.linear_model import LinearRegression
import numpy as np
#import matplotlib.pypplot as plt
import pickle
from xgboost import plot_importance
from script import validation
from script import process_str
from sklearn.model_selection import KFold
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

class deep_predictor:
    def __init__(self, structure_A_num, structure_A_list, structure_B_num, structure_B_list, predictor_type, weight_list):
        self.structure_A_num = structure_A_num
        self.structure_A_list = structure_A_list
        self.structure_B_num = structure_B_num
        self.structure_B_list = structure_B_list
        self.predictor_type = predictor_type
        self.weight_list = weight_list
    def predict(self, X_predict):
        Y_B_predict=[]
        Y_B_predictors_predict = []
        Y_predict_features = []
        Y_A_predict=[]
        for i in range(len(X_predict)):
            Y_A_predict.append([])

        for predictor in self.structure_A_list:
            temp_y = predictor.predict(X_predict)
            for i in range(len(Y_A_predict)):
                Y_A_predict[i].append(temp_y[i])
        for i in range(len(Y_A_predict)):
            for j in X_predict[i]:
                Y_A_predict[i].append(j)
        for predictor in self.structure_B_list:
            Y_B_predictors_predict.append(predictor.predict(Y_A_predict))
        for i in range(len(Y_A_predict)):
            result_count_dic = {}
            predicting_result_sum = 0
            weight_sum = 0
            for index in range(len(Y_B_predictors_predict)):
                if self.predictor_type=='regression':
                    predicting_result_sum += (self.weight_list[index]*Y_B_predictors_predict[index][i])
                else:
                    if Y_B_predictors_predict[index][i] not in result_count_dic.keys():
                        result_count_dic[Y_B_predictors_predict[index][i]] = self.weight_list[index]
                    else:
                        result_count_dic[Y_B_predictors_predict[index][i]] += self.weight_list[index]
                weight_sum += self.weight_list[index]
            predicting_result_sum /= weight_sum
            max_count = 0
            max_label = 0
            for key in result_count_dic.keys():
                if max_count < result_count_dic[key]:
                    max_count = result_count_dic[key]
                    max_label = key
            if self.predictor_type=='classfication':
                Y_B_predict.append(max_label)
            else:
                Y_B_predict.append(predicting_result_sum)
        return Y_B_predict

def train(X_train, Y_train, predictor_type, method_param_list, method_param_list2):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
   
    structure_A_num = 0
    structure_A_list = []
    structure_B_num = 0
    structure_B_list = []
    training_X_data2 = []
    training_Y_data2 = [] 


    dataset_len = len(X_train)
    split_len = int(dataset_len/10)+1
    split_X_data = []
    split_Y_data = []
    features_data = []
    index = -1
    for i in range(dataset_len):
        if i%split_len==0:
            index += 1
            split_X_data.append([])
            split_Y_data.append([])
        split_X_data[index].append(X_train[i])
        split_Y_data[index].append(Y_train[i])
    index_s = 0
    index_e = 0
    for index in range(10):
        training_X_data = []
        training_Y_data = []
        for i in range(len(split_Y_data[index])):
            training_X_data2.append([])
            training_Y_data2.append(split_Y_data[index][i])
            index_e += 1
        for i in range(10):
            if index!=i:
                for x in split_X_data[i]:
                    training_X_data.append(x)
                for y in split_Y_data[i]:
                    training_Y_data.append(y)
        for mp_key in method_param_list.keys():
            print('train A '+mp_key)
            real_mp_key = mp_key.split('-')[0]
            if real_mp_key=='xgboost':
                predictor = xgb.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='linear':
                predictor = linear.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='svm':
                predictor = svm.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='decision_tree':
                predictor = decision_tree.train(training_X_data, training_Y_data, predictor_type,method_param_list[mp_key])
            elif real_mp_key=='knn':
                predictor = knn.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='gradient_boosting':
                predictor = gradient_boosting.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='random_forest':
                predictor = random_forest.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='extra_tree':
                predictor = extra_tree.train(training_X_data, training_Y_data, predictor_type, method_param_list[mp_key])
            elif real_mp_key=='lasso_glm':
                predictor = lasso_glm.train(X_train, Y_train, predictor_type, method_param_list[mp_key])
            elif reall_mp_key=='mlp':
                predictor = mlp.train(X_train, Y_train, predictor_type, method_param_list[mp_key])

            temp_prediction = predictor.predict(split_X_data[index])
            for i in range(index_s, index_e):
                training_X_data2[i].append(temp_prediction[i-index_s])
        index_s = index_e
    for index in range(len(X_train)):
        for x in X_train[index]:
            training_X_data2[index].append(x)
    weight_list = []
    for mp_key in method_param_list2.keys():
        print('train B '+mp_key)
        real_mp_key = mp_key.split('-')[0]
        if real_mp_key=='xgboost':
            predictor = xgb.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='linear':
            predictor = linear.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='svm':
            predictor = svm.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='decision_tree':
            predictor = decision_tree.train(training_X_data2, training_Y_data2, predictor_type,method_param_list2[mp_key])
        elif real_mp_key=='knn':
            predictor = knn.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='gradient_boosting':
            predictor = gradient_boosting.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='random_forest':
            predictor = random_forest.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='extra_tree':
            predictor = extra_tree.train(training_X_data2, training_Y_data2, predictor_type, method_param_list2[mp_key])
        elif real_mp_key=='lasso_glm':
            predictor = lasso_glm.train(X_train, Y_train, predictor_type, method_param_list2[mp_key])
        elif reall_mp_key=='mlp':
            predictor = mlp.train(X_train, Y_train, predictor_type, method_param_list2[mp_key])
        structure_B_list.append(predictor)
        weight_list.append(1)
    structure_B_num = len(structure_B_list)
    for mp_key in method_param_list.keys():
        real_mp_key = mp_key.split('-')[0]
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
        structure_A_list.append(predictor)
    structure_A_num = len(structure_A_list)
    dp = deep_predictor(structure_A_num, structure_A_list, structure_B_num, structure_B_list, predictor_type, weight_list)
    return dp
