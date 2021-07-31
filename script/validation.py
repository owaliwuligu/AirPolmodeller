#!/usr/bin/env python
#_*_coding:utf-8_*_
import re,sys,os
import numpy as np
import xgboost as xgb
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
from ml import uni_method
from sklearn.preprocessing import label_binarize
from script import draw_plot

def validate(X, Y, folds, ml_method, output_folder_path, params, param_value_bo, validation_type='regression'):
#   define the validation path
    if output_folder_path[len(output_folder_path)-1]=='/':
        length = len(output_folder_path)
        output_folder_path0 = output_folder_path[:length-1]
    validation_path = output_folder_path + '/' + 'validation_result.txt'
    independant_validation_path = output_folder_path + '/' + 'independant_validation_result.txt'

#   judge the validation type
    judge_temp = Y[0]
#    validation_type = 'classification'
#    for y in Y:
#        if re.search('\.', str(y))!=None:
#            validation_type = 'regression'
#            break

#    if re.search('\.', str(judge_temp))!=None:
#        validation_type = 'regression'   
    mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre, cm, labels = k_fold(np.array(X), np.array(Y), folds, ml_method, validation_type, params, param_value_bo)
    print("kfold completed")
    mae_in, evs_in, mse_in, r2s_in, acc_in, rec_in, f1s_in, mcc_in, fpr_in, tpr_in, pre_in, cm_in, labels_in = independant_validation(np.array(X), np.array(Y), 0.7, ml_method, validation_type, params, param_value_bo)
    print("independant completed")
    Y_predicted, Y_observed = predictedVSreal(np.array(X), np.array(Y), ml_method, validation_type, params, param_value_bo)
    print("predictedVSreal completed")
    draw_plot.draw_scatterplot2(Y_predicted, Y_observed, output_folder_path)
    print("scatter plot completed")
    if validation_type=='regression':
        print("regression!")
        f_out = open(validation_path, 'w')
        if folds!='leave-one-out':
            f_out.write(folds+' folds validation:\nMean Absolute Error:\t'+str(mae).split('.')[0]+'.'+str(mae).split('.')[1][:4]+'\n')
        else:
            f_out.write('Leave-one-out validation:\nMean Aboslute Error:\t'+str(mae).split('.')[0]+'.'+str(mae).split('.')[1][:4]+'\n')
        f_out.write('Mean Squared Error:\t'+str(mse).split('.')[0]+'.'+str(mse).split('.')[1][:4]+'\n')
        f_out.write('Expained Variance Score:\t'+str(evs).split('.')[0]+'.'+str(evs).split('.')[1][:4]+'\n')
        if len(str(r2s).split('.'))>1:
            f_out.write('R^2 Score:\t'+str(r2s).split('.')[0]+'.'+str(r2s).split('.')[1][:4]+'\n')
        f_out.close()
        
        f_out = open(independant_validation_path, 'w') 
#        if folds!='leave-one-out':
#            f_out.write(folds+' folds validation:\nMean Absolute Error:\t'+str(mae).split('.')[0]+'.'+str(mae).split('.')[1][:4]+'\n')
#        else:
        f_out.write('Independent validation:\nMean Aboslute Error:\t'+str(mae_in).split('.')[0]+'.'+str(mae_in).split('.')[1][:4]+'\n')
        f_out.write('Mean Squared Error:\t'+str(mse_in).split('.')[0]+'.'+str(mse_in).split('.')[1][:4]+'\n')
        f_out.write('Expained Variance Score:\t'+str(evs_in).split('.')[0]+'.'+str(evs_in).split('.')[1][:4]+'\n')
        f_out.write('R^2 Score:\t'+str(r2s_in).split('.')[0]+'.'+str(r2s_in).split('.')[1][:4]+'\n')
        f_out.close()
    else:
        print("classification!")
        f_out = open(validation_path, 'w')
        if folds!='leave-one-out':
            f_out.write(folds+' folds validation:\naccuracy:\t'+str(acc).split('.')[0]+'.'+str(acc).split('.')[1][:4]+'\n')
        else:
            f_out.write('leave-one-out validation:\naccuracy:\t'+str(acc.split('.')[0]+'.'+str(acc).split('.')[1][:4])+'\n')
        f_out.write('recall_score:\t'+str(rec).split('.')[0]+'.'+str(rec).split('.')[1][:4]+'\n')
        f_out.write('f1_score:\t'+str(f1s).split('.')[0]+'.'+str(f1s).split('.')[1][:4]+'\n')
        f_out.write('matthews_corrcoef:\t'+str(mcc).split('.')[0]+'.'+str(mcc).split('.')[1][:4]+'\n')
        f_out.write('specifity:\t'+str(tpr).split('.')[0]+'.'+str(tpr).split('.')[1][:4]+'\n')
        f_out.write('precision:\t'+str(pre).split('.')[0]+'.'+str(pre).split('.')[1][:4]+'\n')

        f_out = open(independant_validation_path, 'w')
#        if folds!='leave-one-out':
#            f_out.write(folds+' folds validation:\naccuracy:\t'+str(acc).split('.')[0]+'.'+str(acc).split('.')[1][:4]+'\n')
#        else:
        f_out.write('Independent validation:\naccuracy:\t'+str(acc_in).split('.')[0]+'.'+str(acc_in).split('.')[1][:4]+'\n')
        f_out.write('recall_score:\t'+str(rec_in).split('.')[0]+'.'+str(rec_in).split('.')[1][:4]+'\n')
        f_out.write('f1_score:\t'+str(f1s_in).split('.')[0]+'.'+str(f1s_in).split('.')[1][:4]+'\n')
        f_out.write('matthews_corrcoef:\t'+str(mcc_in).split('.')[0]+'.'+str(mcc_in).split('.')[1][:4]+'\n')
        f_out.write('specifity:\t'+str(tpr_in).split('.')[0]+'.'+str(tpr_in).split('.')[1][:4]+'\n')
        f_out.write('precision:\t'+str(pre_in).split('.')[0]+'.'+str(pre_in).split('.')[1][:4]+'\n')
        f_out.close()
#        f_out.write('confusion matrix:\n')
#        for lb in labels:
#            f_out.write(str(lb)+'\t')
#        f_out.write('\n')
#        for cm_x in range(len(cm)):
#            for cm_y in cm[cm_x]:
#                f_out.write(str(cm_y)+'\t')
#            f_out.write(str(labels[cm_x])+'\n')
#        f_out.write('\n')
#        for i in cm_origin[0]:
#            f_out.write(str(i)+'\t')
#        f_out.write('\n')
#        for i in cm_origin[1]:
#            f_out.write(str(i)+'\t')
        f_out.close()

    #draw ROC Curve
    if validation_type=='classification' and ml_method in ['adaboost', 'gradient_boosting', 'svm', 'decision_tree', 'knn', 'random_forest', 'extra_tree', 'xgboost']:
        draw_plot.draw_roc_curve(np.array(X), np.array(Y), ml_method, params, param_value_bo, output_folder_path, validation_type)

def independant_validation(X, Y, threshold, ml_method, validation_type, params, param_value_bo):
    labels = []
    for yy in Y:
        if yy not in labels:
            labels.append(yy)
    mae = 0.0
    evs = 0.0
    mse = 0.0
    r2s = 0.0
    acc = 0.0
    rec = 0.0
    f1s = 0.0
    mcc = 0.0
    fpr = 0.0
    tpr = 0.0
    pre = 0.0
    cm=[] #ignore cm
    #split the dataset by the shreshold
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    dataset_length = len(X)
    training_set_length = int(dataset_length*threshold)
    testing_set_length = dataset_length - training_set_length
    for i in range(training_set_length):
        train_X.append(X[i])
        train_Y.append(Y[i])
    for j in range(training_set_length, dataset_length):
        test_X.append(X[j])
        test_Y.append(Y[j])

    if validation_type=='regression':
        predictor, ignore_result = uni_method.train(ml_method, train_X, train_Y, validation_type, params, param_value_bo)
        Y_predict = uni_method.predict(predictor, test_X)

        mae += mean_absolute_error(test_Y, Y_predict)
        evs += explained_variance_score(test_Y, Y_predict)
        mse += mean_squared_error(test_Y, Y_predict)
        r2s += r2_score(test_Y, Y_predict)
    if validation_type=='classification':
        predictor, ignore_result = uni_method.train(ml_method, train_X, train_Y, validation_type, params, param_value_bo)
        Y_predict = uni_method.predict(predictor, test_X)

        Y_predict = label_binarize(Y_predict, labels)
        test_Y = label_binarize(test_Y, labels)
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
            for j in range(test_Y.shape[0]):
                if test_Y[j][i]==1 and Y_predict[j][i]==1:
                    temp_tp+=1
                elif test_Y[j][i]==0 and Y_predict[j][i]==0:
                    temp_tn+=1
                elif test_Y[j][i]==1 and Y_predict[j][i]==0:
                    temp_fn+=1
                elif test_Y[j][i]==0 and Y_predict[j][i]==1:
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
        rec = (temp_recall/n_classes)
        acc = (temp_acc_multi_binary/n_classes)
        fpr = (temp_fpr/n_classes)
        mcc = (temp_mcc/n_classes)
        tpr = (temp_tpr/n_classes)
        pre = (temp_precision/n_classes)
        if pre+rec>0:
            f1s = (2*pre*rec)/(pre+rec)
    return mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre, cm, labels

def predictedVSreal(X, Y, ml_method, validation_type, params, param_value_bo):
    k = 10
    kf = KFold(k, True)
    kf.get_n_splits(X)
    Y_predicted = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        if validation_type=='regression':
            predictor, ignore_result = uni_method.train(ml_method, X_train, Y_train, validation_type, params, param_value_bo)
            Y_predict = uni_method.predict(predictor, X_test)

        if validation_type=='classification':
            predictor, ignore_result = uni_method.train(ml_method, X_train, Y_train, validation_type, params, param_value_bo)
            Y_predict = uni_method.predict(predictor, X_test)

        for yyy in Y_predict:
           Y_predicted.append(yyy)
    return Y_predicted, Y     
    
def k_fold(X, Y, folds, ml_method, validation_type, params, param_value_bo):
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
    cm = [] #confusion matrix
#    cm_origin = [[0,0,0,0,0,0], [0,0,0,0,0,0]]  #delete this
    #init cm
    for i in range(len(labels)):
        temp = []
        for j in range(len(labels)):
            temp.append(0)
        cm.append(temp)
    cm_dic = {}
    for i in range(len(labels)):
        cm_dic[labels[i]] = i
#    thresholds = 0.0
    pre = 0.0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        #delete this
#        Y_origin = []
#        for xx in X_test:
#            Y_origin.append(xx[0])
#            print(xx[0])
#        X_test_temp = X_test
#        X_test = []
#        for xx in X_test_temp:
#            temp = []
#            for xxx in range(1, len(xx)):
#                temp.append(xx[xxx])
#            X_test.append(temp)
#
#        X_train_temp = X_train
#        X_train = []
#        for xx in X_train_temp:
#            temp = []
#            for xxx in range(1, len(xx)):
#                temp.append(xx[xxx])
#            X_train.append(temp)
       #delete this


        if validation_type=='regression':
            predictor, ignore_result = uni_method.train(ml_method, X_train, Y_train, validation_type, params, param_value_bo)
            Y_predict = uni_method.predict(predictor, X_test)
            
            mae += mean_absolute_error(Y_test, Y_predict)
            evs += explained_variance_score(Y_test, Y_predict)
            mse += mean_squared_error(Y_test, Y_predict)
            r2s += r2_score(Y_test, Y_predict)
        if validation_type=='classification':
            predictor, ignore_result = uni_method.train(ml_method, X_train, Y_train, validation_type, params, param_value_bo)
            #binarize the labels
            Y_predict = uni_method.predict(predictor, X_test)
            
            #calculate the confusion matrix
            for cm_i in range(len(Y_predict)):
                cm[cm_dic[Y_test[cm_i]]][cm_dic[Y_predict[cm_i]]] += 1
#                if Y_predict[cm_i]==0:
#                    cm_origin[0][int(Y_origin[cm_i])]+=1
#                elif Y_predict[cm_i]==1:
#                    cm_origin[1][int(Y_origin[cm_i])]+=1

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
#            acc += precision_score(Y_test, Y_predict, average='macro')
#            rec += recall_score(Y_test, Y_predict, average='macro')
#            f1s += f1_score(Y_test, Y_predict, average='weighted')
#            mcc += matthews_corrcoef(Y_test, Y_predict)
#            fpr0, tpr0, thresholds0 = roc_curve(Y_test, Y_predict)
#            fpr += fpr0
#            tpr += tpr0
#            thresholds0 += thresholds0
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
    return mae, evs, mse, r2s, acc, rec, f1s, mcc, fpr, tpr, pre, cm, labels
