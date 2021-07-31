#/usr/bin/env python
#_*_coding:utf-8_*_
import argparse
import sys
from script import check_args
from script import read_code
from script import validation
from script import save_file
from script import process_str
from script import draw_plot
from ml import uni_method
import MySQLdb

parser = argparse.ArgumentParser(usage="It's usage tip.")
parser.add_argument('--mode', type=str, help='[train|predict]: \'train\' mode helps user train his own models and \'predict\' mode makes preiction with the model and data provided by user')
parser.add_argument('--input_train', type=str, help='The path of training file. It is necessary under \'train\' mode')
parser.add_argument('--input_train_sep', type=str, choices=['comma', 'tab'], default='tab', help='The separator of the training file. The default value is \'tab\'.')
parser.add_argument('--input_predict_sep', type=str, choices=['comma', 'tab'], default='tab', help='The separator of the predicting file. The default value is \'tab\'.')
parser.add_argument('--input_predict', type=str, help='The path of prediction file. It is necessary under \'predict\' mode and optional under \'train\' mode')
parser.add_argument('--model', type=str, help='The path of model. It is necessary under \'predict\' mode')
parser.add_argument('--kfold', type=str, help='[5|10|leave-one-out]: this argument is optional under \'train\' model, and it represents the way of cross validation.')
parser.add_argument('--output_path', type=str, help='The folder path for saving the result files.')
#parser.add_argument('--validation_path', type=str, help='The path for saving validation result. It is necessary if the \'kfold\' was set.')
#parser.add_argument('--output_model', type=str, help='The path of output model.')
#parser.add_argument('--output_result', type=str, help='The path of the predictioin result')
parser.add_argument('--features', type=str, help='The column id list of features. It is optional under \'train\' mode. The default list contains all the columns except the last column. e.g. --features 0,1,3,5 or --features 0-10')
parser.add_argument('--label', type=int, help='The column id of label. It is optional under \'train\' mode. The default value is the id of the last column. e.g. --label 6')
parser.add_argument('--ml', type=str, help='[lasso_glm|xgboost|svm|linear|logistic|decision_tree|knn|adaboost|gradient_boosting|random_forest|sklearn_baggin|extra_tree|bagging|boosting|stacking|deep_ensemble]: The machine learning algorightm.')
parser.add_argument('--params', type=str, help='Set the parameter for the algorithm. e.g. --params max_depth:5,gamma:0,booster:gbtree')
parser.add_argument('--ml_type', type=str, help='[classification|regression]')
parser.add_argument('--scatter', type=str, help='Set pairs of features to generate scatterplots. e.g. --scatter 1:2,4:5')
parser.add_argument('--distribution', type=str, help='Choose the columns for which you want to get distribution plots' )
parser.add_argument('--jobId', type=str, required=True, help='the job id of the programme')
args = parser.parse_args()

#check the arguments
check_args.check_args(args)

#generate dictionary in temperary folder
dictionary_path,y_dictionary_path = read_code.create_dictionary(args.output_path)

#set the default ml_type
if args.ml_type==None:
    args.ml_type = 'regression'

if args.input_train_sep=='comma':
    args.input_train_sep = ','
else:
    args.input_train_sep = '\t'

if args.input_predict_sep=='comma':
    args.input_predict_sep = ','
else:
    args.input_predict_sep = '\t'


#draw scatter
path = args.input_train
sep = args.input_train_sep
if args.input_train==None:
    path = args.input_predict
    sep = args.input_predict_sep
X, headlines, X_original = read_code.read_columns(path, dictionary_path, sep)
if args.scatter!=None:
    draw_plot.draw_scatterplot(X, headlines, args.scatter, args.output_path)

#draw distribution plot
if args.distribution!=None:
    draw_plot.draw_barplot(X_original, headlines, args.distribution, args.output_path)

if args.mode=='train':
    if args.features!=None or args.label!=None:
        feature_list, label = process_str.get_features_label(args.features, args.label)
        X,Y,headlines = read_code.read_training_file(args.input_train, dictionary_path, y_dictionary_path, args.input_train_sep, feature_list, label)
    else:
        X,Y,headlines = read_code.read_training_file(args.input_train, dictionary_path, y_dictionary_path, args.input_train_sep)
    print("read features")
    #train model
    if args.params!=None:
        clf, param_value_bo = uni_method.train(args.ml, X, Y, args.ml_type, args.params)
    else:
        clf, param_value_bo = uni_method.train(args.ml, X, Y, args.ml_type)
    print('train model')
    #validation
    if args.kfold!=None:
        validation.validate(X, Y, args.kfold, args.ml, args.output_path, args.params, param_value_bo, args.ml_type)
    print('validated!')
    #save model
    if args.ml=='bagging':
        flag = 1
    else:
        flag = 0
    save_file.save_model(clf, args.output_path, flag, args.input_predict)
    print('saved model!')
    if args.input_predict != None:
        X_pre, headlines = read_code.read_prediction_file(args.input_predict, dictionary_path, args.input_predict_sep)
        #predict
        Y_pre = uni_method.predict(clf, X_pre)
        #read the X_original_pre
        X_original_pre = read_code.read_original(args.input_predict, args.input_predict_sep)
        #save result
        save_file.save_prediction_result(X_original_pre, Y_pre, headlines, args.output_path, args.output_path)
else:
    #load model
    predictor, dic_folder_path = uni_method.load_model(args.model)
    #read predictionn file
    X_pre, headlines = read_code.read_prediction_file(args.input_predict, dic_folder_path+'models_dir_3A89BDdsiwa83Eaeek/dictionary', args.input_predict_sep)
    #read the X_original_pre
    X_original_pre = read_code.read_original(args.input_predict, args.input_predict_sep)
    #predict
    Y_pre = uni_method.predict(predictor, X_pre)
    #save result
    save_file.save_prediction_result(X_original_pre, Y_pre, headlines, args.output_path, dic_folder_path)
save_file.delete_temp(args.output_path)

db = MySQLdb.connect("localhost", "PASSION", "PASSION", "AirPollution", charset='utf8')
cursor = db.cursor()
cursor.execute('UPDATE joblist SET job_status=\'Completed\' where job_id=\''+args.jobId+'\';')
db.commit()
db.close()
f = open('/flagship-data/jinxiang/DeepCleave/AirPollution/jobNum.txt','r')
num = f.read()
f.close()
f = open('/flagship-data/jinxiang/DeepCleave/AirPollution/jobNum.txt', 'w')
f.write(str(int(num)-1))
f.close()

