#!/usr/bin/env python
#_*_coding:utf-8_*_
import sys,os,re
from script import check_file

def check_args(args):
    if not args.mode:
        print('Please set the mode!\n')
        sys.stderr.write('>Please set the mode!\n')
        sys.exit(1)
    else:
        if args.mode!='train' and args.mode!='predict':
            print('Please set the mode as \'train\' or \'predict\', \neg: --mode train\n')
            sys.stderr.write('>Please set the mode as \'train\' or \'predict\', \neg: --mode train\n')
            sys.exit(1)
        elif args.mode=='train':
            if not args.input_train:
                print('Please set the path of training file!\neg: --input_train example/training.txt\n')
                sys.stderr.write('>Please set the path of training file!\neg: --input_train example/training.txt\n')
                sys.exit(1)
            #check the format of training file
            wrong_msg = check_file.check_training_file(args.input_train)
            if wrong_msg!='':
                print(wrong_msg)
                sys.stderr.write('>'+wrong_msg)
                sys.exit(1)
            #check the format of prediction file
            if args.input_predict!=None:
                wrong_msg = check_file.check_prediction_file(args.input_predict)
                if wrong_msg!='':
                    print(wrong_msg)
                    sys.stderr.write('>'+wrong_msg)
                    sys.exit(1)
#                if not args.output_result:
#                    print('Please set the path of the prediction result, \neg: --output_result example/')
#                    sys.stedrr.write('>Please set the path of the prediction result, \neg: --output_result example/')
#                    sys.exit(1)
#                if not os.path.exists(output_result):
#                    print('There\'s no directory in '+output_result+'!\n')
#                    sys.stderr.write('>There\'s no directory in '+output_result+'!\n')
#                    sys.exit(1)
#            if args.output_model!=None:
#                if not os.path.exists(args.output_model):
#                    print('The path for the output model is invalid, please correct it!\n')
#                    sys.stderr.write('>The path for the output model is invalid, please correct it\n')
#                    sys.exit(1)
            if args.output_path==None:
                print('Please set the output path!\n')
                sys.stderr.write('>Please set the output path!\n')
                sys.exit(1)
            if not os.path.exists(args.output_path):
                print('The output path is not exsists!\n')
                sys.stderr.write('>The output path is not exsists!\n')
                sys.exit(1)
            if args.features!=None:
                flag0 = True
                if re.search('-', args.features)!=None:
                    temp_list = args.features.split('-')
                    if len(temp_list)!=2:
                        flag0 = False
                    flag = True
                    for i in temp_list[0]:
                        if ord(i)<ord('0') or ord(i)>ord('9'):
                            flag0 = False
                            break
                    for j in temp_list[1]:
                        if ord(j)<ord('0') or ord(j)>ord('9'):
                            flag0 = False
                            break
                else:
                    temp_list = args.features.split(',')
                    if len(temp_list)<1:
                         flag0 = False
                    for i in temp_list:
                        for j in i:
                            if ord(j)<ord('0') or ord(j)>ord('9'):
                                flag0 = False
                                break
                        if flag0==False:
                            break
                if flag0==False:
                    print('Please set the features as correct format, \neg: --features 0-10 or --features 0,1,3,5\n')
                    sys.stderr.write('>The set the features as correct format, \neg: --features 0-10 or --features 0,1,3,5\n')
                    sys.exit(1)
            if args.label!=None:
                flag = True
                for i in str(args.label):
                    if ord(i)<ord('0') or ord(i)>ord('9'):
                        print('Please set the label as correct format, \neg: --label 6\n')
                        sys.stderr.write('>Please set the label as correct format, \neg: --label 6\n')
                        sys.exit(1)
            if not args.ml:
                print('Please set the machine learning algorithm, \neg: --ml xgboost')
                sys.stderr.write('>Please set the machine learning algorithm, \neg: --ml xgboost')
                sys.exit(1)
            else:
                if args.ml not in ['mlp', 'lasso_glm', 'xgboost', 'svm', 'linear', 'logistic', 'decision_tree', 'knn', 'adaboost', 'gradient_boosting', 'random_forest', 'sklearn_bagging', 'extra_tree', 'bagging', 'boosting', 'stacking', 'deep_ensemble']:
                    print('Please set the machine learning algorithm in [mlp|lasso_glm|xgboost|svm|linear|logistic|decision_tree|knn|adaboost|gradient_boosting|random_forest|sklearn_bagging|extra_tree|bagging|boosting|stacking|deep_ensemble], \neg: --ml xgboost')
                    sys.stderr.write('>Please set the machine learning algorithm in [mlp|lasso_glm|xgboost|svm|linear|logistic|decision_tree|knn|adaboost|gradient_boosting|random_forest|sklearn_bagging|extra_tree|bagging|boosting|stacking|deep_ensemble], \neg: --ml xgboost')
                    sys.exit(1)
            if args.params!=None:
                if args.params!='BO':
                    if args.ml not in ['bagging', 'stacking', 'deep_ensemble']:
                        param_list = args.params.split(',')
                        for param in param_list:
                            if len(re.findall(':', param))!=1:
                                print('Please check the format of value for parameter \'params\'!\n')
                                sys.stderr.write('>Please check the format of value for parameter \'params\'!\n')
                                sys.exit(1)
                    else:
                        if args.ml=='deep_ensemble':
                            method_param_list = args.params.split('I')[0].split('#') + args.params.split('I')[1].split('#')
                        else:
                            method_param_list = args.params.split('#')
                        for mp in method_param_list:
                            if len(mp.split('@'))>2:
                                print('Please check the format of the value for parameter \'params\'!\n')
                                sys.stderr.write('>Please check the format of the value for parameter \'params\'!\n')
                                sys.exit(1)
                            method_temp = mp.split('@')[0]
                            if re.search('\*', method_temp)!=None:
                                method_temp = method_temp[:len(method_temp)-1]
                            if method_temp not in ['xgboost', 'svm', 'linear', 'logistic', 'decision_tree', 'knn', 'adaboost', 'gradient_boosting', 'random_forest', 'extra_tree', 'mlp', 'lasso_glm']:
                                print('Please check the format of the value for parameter \'params\'!\n')
                                sys.stderr.write('>Please check the format of the value for parameter \'params\'!\n')
                                sys.exit(1)
                            if len(mp.split('@'))==1:
                                continue
                            params_temp = mp.split('@')[1]
                            if params_temp=='BO':
                                continue
                            param_list = params_temp.split(',')
                            for param in param_list:
                                if len(re.findall(':',param))!=1:
                                    print('Please check the format of value for parameter \'params\'!\n')
                                    sys.stderr.write('>Please check the format of value for parameter \'params\'!\n')
                                    sys.exit(1)
            if args.ml_type!=None:
                if args.ml_type not in ['classification', 'regression']:
                    print('Please check ml_type!\n')
                    sys.stderr.write('>Please check ml_type!\n')
                    sys.exit(1)
            if args.scatter!=None:
                pair_list = args.scatter.split(',')
                for pair in pair_list:
                    if len(re.findall(':', pair))!=1:
                        print('Please check the format of value for parameter \'scatter\'!\n')
                        sys.stderr.write('>Please check the format of value for parameter \'scatter\'!\n')
                        sys.exit(1)
                    temp_list = pair.split(':')
                    for temp in temp_list:
                        for i in temp:
                            if ord(i)<ord('0') or ord(i)>ord('9'):
                                break
        else:
            if not args.input_predict:
                print('Please set the path of prediction file, \neg: --input_predict example/prediction.txt\n')
                sys.stderr.write('>Please set the path of prediction file, \neg: --input_predict example/prediction.txt\n')
                sys.exit(1)
            else:
                wrong_msg = check_file.check_prediction_file(args.input_predict)
                if wrong_msg!='':
                    print(wrong_msg)
                    sys.stderr.write(wrong_msg)
                    sys.exit(1)
#                if not args.output_result:
#                    print('Please set the path of the prediction result, \neg: --output_result example/')
#                    sys.stedrr.write('>Please set the path of the prediction result, \neg: --output_result example/')
#                    sys.exit(1)
#                if not os.path.exists(args.output_result):
#                    print('There\'s no directory in '+args.output_result+'!\n')
#                    sys.stderr.write('>There\'s no directory in '+args.output_result+'!\n')
#                    sys.exit(1)
                if not args.model:
                    print('Please set the path of model, \neg: --model example/xgboot.model')
                    sys.stderr.write('>Please set the path of model, \neg: --model example/xgboot.model')
                    sys.exit(1)
                elif not os.path.exists(args.model):
                    print('There\'s no such file: '+ args.model+'!\n')
                    sys.stderr.write('>There\'s no such file: '+args.model+'!\n')
                    sys.exit(1)
                if args.output_path==None:
                    print('Please set the output path!\n')
                    sys.stderr.write('>Please set the output path!\n')
                    sys.exit(1)
                if not os.path.exists(args.output_path):
                    print('The output path is not exsists!\n')
                    sys.stderr.write('>The output path is not exsists!\n')
                    sys.exit(1)
                if args.scatter!=None:
                    pair_list = args.scatter.split(',')
                    for pair in pair_list:
                        if len(re.findall(':', pair))!=1:
                            print('Please check the format of value for parameter \'scatter\'!\n')
                            sys.stderr.write('>Please check the format of value for parameter \'scatter\'!\n')
                            sys.exit(1)
                        temp_list = pair.split(':')
                        for temp in temp_list:
                            for i in temp:
                                if ord(i)<ord('0') or ord(i)>ord('9'):
                                    break
#        if not args.ml:
#            print('Please set the machine learning algorithm, \neg: --ml xgboost')
#            sys.stderr.write('>Please set the machine learning algorithm, \neg: --ml xgboost')
#            sys.exit(1)
#        else:
#            if args.ml not in ['xgboost', 'svm', 'linear', 'dt', 'knn', 'adaboost', 'gbrt', 'rf']:
#                print('Please set the machine learning algorithm in [xgboost|svm|linear|dt|knn|adaboost|gbrt|rf], \neg: --ml xgboost')
#                sys.stderr.write('>Please set the machine learning algorithm in [xgboost|svm|linear|dt|knn|adaboost|gbrt|rf], \neg: --ml xgboost')
#                sys.exit(1)
