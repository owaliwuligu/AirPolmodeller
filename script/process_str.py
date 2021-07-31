#!/usr/bin/env python
#_*_coding:utf-8_*_
import re

#check if the string contains letters
def my_check_letter(string):
    for s in string:
        if (ord(s)-ord('0')<0 or ord(s)-ord('0')>9) and s!='.' and s!='-':
            return True
    return False

def check_letter(string):
    try:
        float(string)
        return False
    except:
        return True

#transform a string to a hash code
def djb_hash(string):
    base = 5381

    for s in string:
        base = ((base<<5) + base) + ord(s)
    base = base & (~(1<<31))
    return base

#split the string and get the list of features
def get_features_label(features_str, label_str):
    label = -1
    if label_str!=None:
        label = int(label_str)
    
    #split the string of features
    if features_str==None:
        feature_list = None
    elif re.search('-', features_str):
        f_list = features_str.split('-')
        feature_list = range(int(f_list[0]), int(f_list[1])+1)
    else:
        f_list = features_str.split(',')
        feature_list = []
        for f in f_list:
            feature_list.append(int(f))
    return feature_list, label
