#/usr/bin/env python
#_*_coding:utf-8_*_
import sys,re,os
import numpy as np
import matplotlib.pyplot as plt
from script import process_str
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from ml import uni_method
#from scipy.optimize import leastsq
import math
def draw_roc_curve(X, Y, ml_method, params, param_value_bo, path, predictor_type):
    label_list = []
    for y in Y:
        if y not in label_list:
            label_list.append(y)
    Y_binary = label_binarize(Y, label_list)
    n_classes = Y_binary.shape[1]
    
    #split training & testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=.5, random_state=1234)
    predictor, _ = uni_method.train(ml_method, X_train, Y_train, predictor_type, params, param_value_bo, OneVsRest=True)
    Y_score = predictor.predict_proba(X_test)
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
#    print(X_test)
#    print(Y_score[:, 0])
    for i in range(n_classes):
        fpr_temp, tpr_temp, _=roc_curve(Y_test[:, i], Y_score[:, i])
        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        roc_auc_temp = auc(fpr_temp, tpr_temp)
        roc_auc_list.append(roc_auc_temp)

    #compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr_list[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr_list[i], tpr_list[i])
    #finally average it and compute AUC
    mean_tpr /= n_classes
    roc_auc = auc(all_fpr, mean_tpr)
    
    #draw the plot
    plt.plot(all_fpr, mean_tpr, label=ml_method + ' (area = {0:0.4f})' ''.format(roc_auc), linewidth=1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0.0, 1.0)
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Averaged ROC')
    plt.legend(loc="lower right")
    plt.savefig(path + '/ROC_Curve.png')

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

#    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b

def draw_scatterplot2(y1, y2, path):
    plt.rcParams['figure.dpi'] = 1000
    plt.scatter(y1, y2, s=10, alpha=0.6)
    plt.xlabel('Predicted value', fontdict={'family':'Times New Roman', 'size':16})
    plt.ylabel('Observed value', fontdict={'family':'Times New Roman', 'size':16})
    a,b = best_fit(y2, y1)
    y_fit = [a+b*xi for xi in y2]
    plt.plot(y2, y_fit)
    if path[len(path)-1]=='/':
        plt.savefig(path+'pVSo.png', dpi=1000)
    else:
        plt.savefig(path+'/pVSo.png', dpi=1000)
    plt.clf()

def draw_scatterplot(X, headlines, pairs, path):
    #split the feature pairs
    pair_list = pairs.split(',')
    for pair in pair_list:
        x_id = pair.split(':')[0]
        y_id = pair.split(':')[1]
        X_sca = []
        Y_sca = []
        for j in range(len(X)):
            X_sca.append(X[j][int(x_id)])
            Y_sca.append(X[j][int(y_id)])
        plt.rcParams['figure.dpi'] = 1000
        plt.scatter(X_sca, Y_sca, s=10, alpha=0.6)
        plt.xlabel(headlines[int(x_id)], fontdict={'family':'Times New Roman', 'size':16})
        plt.ylabel(headlines[int(y_id)], fontdict={'family':'Times New Roman', 'size':16})
        if path[len(path)-1]=='/':
            plt.savefig(path+x_id+'_'+y_id+'.png')
        else:
            plt.savefig(path+'/'+x_id+'_'+y_id+'.png')
        plt.clf()

def draw_barplot(X_original, headlines, selection, path):
    selection_list = selection.split(',')
    for select_id0 in selection_list:
        print(str(select_id0))
        select_id = int(select_id0)
        if select_id >= len(headlines):
            print('The column '+str(select_id)+' you selected does not exist!\n')
            sys.stderr.write('>The column '+str(select_id)+' you selected does not exist!\n')
            sys.exit(1)
        bar_dic = {}
        name_list = []
        num_list = []
        letter_flag = False
        min_label = 999999999
        max_label = -999999999
        if_float = False
        for i in X_original:
            if i[select_id] not in bar_dic.keys():
                if i[select_id] not in ['', 'NA']:
                    bar_dic[i[select_id]] = 1
                    if process_str.check_letter(i[select_id])==True:
                        letter_flag = True
                    else:
                        if re.search('\.', i[select_id])!=None:
                            if_float = True
                            break
                        if int(i[select_id])<min_label:
                            min_label = int(i[select_id])
                        if int(i[select_id])>max_label:
                            max_label = int(i[select_id])
            else:
                bar_dic[i[select_id]] += 1
        if if_float==True:
            print('The value of the column '+str(select_id)+ ' is not integer or string, so it will be skipped!\n')
            continue
        plt.rcParams['figure.dpi'] = 1000
        if letter_flag==False:
            for j in range(min_label, max_label+1):
#                print(str(j))
                if str(j) in bar_dic.keys():
                    num_list.append(bar_dic[str(j)])
                else:
                    num_list.append(0)
            plt.bar(range(min_label, max_label+1), num_list)
        else:
            for key in bar_dic.keys():
                name_list.append(key)
                num_list.append(bar_dic[key])
            plt.bar(range(len(num_list)), num_list, tick_label=name_list)
        plt.savefig(path+'/'+str(select_id)+'_distribution.png')
        plt.clf()
