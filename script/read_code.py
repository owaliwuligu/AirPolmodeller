#!/usr/bin/env python
#_*_coding:utf-8_*_
import re
import os
import sys
import numpy as np
#import process_str
from script import process_str
from script import save_file

def create_dictionary(folder_path):
    if os.path.exists(folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek'):
        print('The folder \'models_dir_3A89BDdsiwa83Eaeek\' has the same name with the temperary folder generating by the programm, please rename or move the folder!\n')
        sys.stderr.write('>The folder \'models_dir_3A89BDdsiwa83Eaeek\' has the same name with the temperary folder generating by the programm, please rename or move the folder!\n')
        sys.exit(1)
    try:
        os.mkdir(folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek')
        f_dic = open(folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek/dictionary', 'w')
        f_dic.close()
        f_ydic = open(folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek/y_dictionary', 'w')
        f_ydic.close()
    except BaseException as e:
        print(e)
        print('Errors ocur during generating the result files, please check the path and permission!\n')
        sys.stderr.write('>Errors ocur during generating the result files, please check the path and permission!\n')
        save_file.delete_temp(folder_path)
        sys.exit(1)
    return folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek/dictionary', folder_path+'/'+'models_dir_3A89BDdsiwa83Eaeek/y_dictionary'

def read_dictionary(dictionary_path):
    f = open(dictionary_path)
    data = f.read()
    f.close()
    data_list = data.splitlines(False)
    dic={}
    for line in data_list:
        if line=='':
            break
        line_list = line.split(' ')
        dic[line_list[1]] = int(line_list[0])
    return dic, len(data_list)

def read_dictionary2(dictionary_path):
    f = open(dictionary_path)
    data = f.read()
    f.close()
    data_list = data.splitlines(False)
    dic = {}
    for line in data_list:
        if line=='':
            break
        line_list = line.split(' ')
        dic[line_list[0]] = line_list[1]
    return dic

def append_dictionary(dictionary_path, dic):
    f = open(dictionary_path, 'a')
    for i in dic:
        f.write(str(dic[i])+' '+i+'\n')
    f.close()

def read_original(file_path, sep):
    f = open(file_path)
    line = f.readline()
    head_list = line.split(sep)
    headlines = []
    for h in head_list:
        h = h.strip('\r')
        h = h.strip('\n')
        headlines.append(h)
    X_ori = []
    while 1:
        lines = f.readlines(3000000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line = line.strip('\r')
            if line=='':
                break
            temp = []
            line_list = line.split(sep)
            flag = True
            for i in line_list:
                if i=='' or i=='NA':
                    flag = False
                    break
                temp.append(i)
            if flag==True:
                X_ori.append(temp)
    f.close()
    return X_ori

def read_columns(file_path, dictionary_path, sep):
    f = open(file_path)
    line = f.readline()
    j_len = len(line.split(sep))
    head_list = line.split(sep)
    headlines = []
    for h in head_list:
        h = h.strip('\r')
        h = h.strip('\n')
        headlines.append(h)
    X=[]
    X_original=[]
#    dic, next_dic_id = read_dictionary()
    dic = {}
    next_dic_id = 0
    while 1:
        lines = f.readlines(3000000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line = line.strip('\r')
            if line=='':
                break
            temp = []
            temp_original = []
            line_list = line.split(sep)
            flag = True
            for i in range(j_len):
                temp_original.append(line_list[i])
            for i in range(j_len):
#                temp_original.append(line_list[i])
                if line_list[i]=='' or line_list[i]=='NA':
                    flag = False
                    break
                if process_str.check_letter(line_list[i]):
                    if line_list[i] in dic.keys():
                        temp.append(dic[line_list[i]])
                    else:
                        temp.append(next_dic_id)
                        dic[line_list[i]] = next_dic_id
                        next_dic_id += 1
                else:
                    if re.search('\.', line_list[i])!=None:
                        temp.append(float(line_list[i]))
                    else:
                        temp.append(int(line_list[i]))
            X_original.append(temp_original)
            if flag==True:
                X.append(temp)

    append_dictionary(dictionary_path, dic)
    f.close()
    return X, headlines, X_original

def read_prediction_file(file_path, dictionary_path, sep):
    f = open(file_path)
    line = f.readline()
    j_len = len(line.split(sep))
    head_list = line.split(sep)
    headlines = []
    for h in head_list:
        h = h.strip('\r')
        h = h.strip('\n')
        headlines.append(h)
    X=[]
    dic, next_dic_id = read_dictionary(dictionary_path)
#    dic = {}
#    next_dic_id = 0
    while 1:
        lines = f.readlines(3000000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line = line.strip('\r')
            if line=='':
                break
            temp = []
            line_list = line.split(sep)
            flag = True
            for i in range(j_len):
                if line_list[i]=='' or line_list[i]=='NA':
                    flag = False
                    break
                if process_str.check_letter(line_list[i]):
                    if line_list[i] in dic.keys():
                        temp.append(dic[line_list[i]])
                    else:
                        temp.append(next_dic_id)
                        dic[line_list[i]] = next_dic_id
                        next_dic_id += 1
                else:
                    if re.search('\.', line_list[i])!=None:
                        temp.append(float(line_list[i]))
                    else:
                        temp.append(int(line_list[i]))
            if flag==True:
                X.append(temp)

    append_dictionary(dictionary_path, dic)
    f.close()
    return X, headlines

def read_training_file(file_path, dictionary_path, y_dictionary_path, sep, feature_list=None, label=-1):
    f = open(file_path)
    line = f.readline()
    j_len = len(line.split(sep))
    head_list = line.split(sep)
    headlines = []
    for h in head_list:
        h = h.strip('\r')
        h = h.strip('\n')
        headlines.append(h)
#    print(j_len)
    X=[]
    Y=[]
    dic, next_dic_id = read_dictionary(dictionary_path)
#    dic = {}
#    next_dic_id = 0
    y_dic = {}
    y_next_dic_id = 0
    if feature_list==None:
        f_list = range(j_len-1)
    else:
        f_list = feature_list
    
    if label==-1:
        y = j_len-1
    else:
        y = label

    while 1:
        lines = f.readlines(3000000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line = line.strip('\r')
            if line=='':
                break
            temp = []
            line_list = line.split(sep)
            flag = True
            for i in f_list:
                if line_list[i]=='' or line_list[i]=='NA':
                    flag = False
                    break
                if process_str.check_letter(line_list[i]):
                    if line_list[i] in dic.keys():
                        temp.append(dic[line_list[i]])
                    else:
                        temp.append(next_dic_id)
                        dic[line_list[i]] = next_dic_id
                        next_dic_id += 1
                else:
                    if re.search('\.', line_list[i])!=None:
                        temp.append(float(line_list[i]))
                    else:
                        temp.append(int(line_list[i]))
            if flag==True:
#                print(line_list[y]+' '+str(process_str.check_letter(line_list[y])))
                if line_list[y]=='' or line_list[y]=='NA':
                    continue
                if process_str.check_letter(line_list[y]):
                    if line_list[y] in y_dic.keys():
                        Y.append(y_dic[line_list[y]])
                        
                    else:
                        Y.append(y_next_dic_id)
                        y_dic[line_list[y]] = y_next_dic_id
                        y_next_dic_id += 1
                else:
                    if re.search('\.', line_list[y])!=None:
                        Y.append(float(line_list[y]))
                    else:
                        Y.append(int(line_list[y]))
                X.append(temp)

    append_dictionary(dictionary_path, dic)
    append_dictionary(y_dictionary_path, y_dic)
    f.close()
    return X,Y,headlines
#read_training_file('../example/air_data_mini.txt')
#print(read_training_file('../example/air_data_mini.txt'))
