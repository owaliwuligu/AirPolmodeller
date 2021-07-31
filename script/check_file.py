#!/usr/bin/env python
#_*_coding:utf-8_*_
import re,sys,os
from script import process_str

def check_training_file(training_file_path):
    if not os.path.exists(training_file_path):
        wrong_msg = 'There\'s no such file: '+training_file_path+'!\n'
        return wrong_msg
    f = open(training_file_path)
    first_line = f.readline()
    first_line_list = first_line.split('\t')
    wrong_msg = ''
    col_num = len(first_line_list)
    if col_num<2:
        wrong_msg = 'Format Error: the number of variables in your training file should be equal or greater than 2!\n'
    else:
        #check the headline
        flag = True
        for i in first_line_list:
            if process_str.check_letter(i)==False:
                flag = False
                break
        if flag==False:
            wrong_msg = 'Format Error: you should add a variable name for each column of data at the first line in your training file!\n'
    
    while 1:
        lines = f.readlines()
        if not lines:
            break
        for line in lines:
            line_list = line.split('\t')
            if len(line_list) != col_num:
                wrong_msg = 'Format Error: please check the format of your training file!\n'
    f.close()
    return wrong_msg

def check_prediction_file(prediction_file_path):
    if not os.path.exists(prediction_file_path):
        wrong_msg = 'There\'s no such file: '+prediction_file_path+'!\n'
        return wrong_msg
    f = open(prediction_file_path)
    first_line = f.readline()
    first_line_list = first_line.split('\t')
    wrong_msg = ''
    col_num = len(first_line_list)
    if col_num<1:
        wrong_msg = 'Format Error: the number of variables in your prediction file should be equal or greater than 1!\n'
    else:
        #check the headline
        flag = True
        for i in first_line_list:
            if process_str.check_letter(i)==False:
                flag = False
                break
        if flag==False:
            wrong_msg = 'Format Error: you should add a variable name for each column of data at the first line in your prediction file!\n'

    while 1:
        lines = f.readlines()
        if not lines:
            break
        for line in lines:
            line_list = line.split('\t')
            if len(line_list) != col_num:
                wrong_msg = 'Format Error: please check the format of your prediction file!\n'
    f.close()
    return wrong_msg

