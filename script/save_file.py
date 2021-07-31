#!/usr/bin/env python
#_*_coding:utf-8_*_
import re,sys,os
import pickle
import tarfile
import shutil
import tarfile
from script import read_code
from script import process_str

def delete_temp(output_folder):
    if os.path.exists(output_folder+'/models_dir_3A89BDdsiwa83Eaeek'):
        shutil.rmtree(output_folder+'/models_dir_3A89BDdsiwa83Eaeek')

def compress_file(tarfilename, dirname):
    if os.path.isfile(dirname):
        with tarfile.open(tarfilename, 'w') as tar:
            tar.add(dirname)
    else:
        with tarfile.open(tarfilename, 'w') as tar:
            for root, dirs, files in os.walk(dirname):
                for single_file in files:
                    filepath = os.path.join(root, single_file)
                    tar.add(filepath)
#if flag equals to 0, it means the predictor is single normal model; if flag equals to 1, it means the predictor is for bagging strategy; if flag ... 2, it ... for boosting strategy; if flag ... 3, it ... for stacking strategy
def save_model(predictor, output_folder_path, flag=0, input_predict=None):
#    if output_folder_path[len(output_folder_path)-1]=='/':
#        length = len(output_folder_path)
#        output_folder_path0 = output_folder_path[:length-1]
#    output_model_path = output_folder_path + '/models_dir_3A89BDdsiwa83Eaeek'
    try:
        #predictor.get_booster().save_model(output_model_path)
        if flag==0:
            path_current = os.getcwd()
            os.chdir(output_folder_path)
            f = open('models_dir_3A89BDdsiwa83Eaeek/model', 'wb')
            pickle.dump(predictor, f)
            f.close()
            f_out = open('models_dir_3A89BDdsiwa83Eaeek/flag', 'w')
            f_out.write('algorithm\n'+'algorithm')
            f_out.close()
            compress_file('model', 'models_dir_3A89BDdsiwa83Eaeek')
            if input_predict==None:
                shutil.rmtree('models_dir_3A89BDdsiwa83Eaeek')
            os.chdir(path_current)
        elif flag==1:
            path_current = os.getcwd()
            os.chdir(output_folder_path)
            output_folder_temp = 'models_dir_3A89BDdsiwa83Eaeek'
            if not os.path.exists('models_dir_3A89BDdsiwa83Eaeek'):
                print('The temperary folder is missed!\n')
                sys.stderr.write('>The temperary folder is missed!\n')
                sys.exit(1)
            for pre_id in range(predictor.model_num):
                output_model_path = 'models_dir_3A89BDdsiwa83Eaeek/model' + str(pre_id)
                f = open(output_model_path, 'wb')
                pickle.dump(predictor.model_list[pre_id], open(output_model_path, 'wb'))
                f.close()
            f_out = open('models_dir_3A89BDdsiwa83Eaeek/flag', 'w')
            f_out.write('bagging\n'+predictor.predictor_type)
            f_out.close()
            compress_file('model', 'models_dir_3A89BDdsiwa83Eaeek')
            if input_predict==None:
                shutil.rmtree('models_dir_3A89BDdsiwa83Eaeek')
            os.chdir(path_current)
        elif flag==2:
            #deep_ensemble
            path_current = os.getcwd()
            os.chdir(output_folder_path)
            output_folder_temp = 'models_dir_3A89BDdsiwa83Eaeek'
            if not os.path.exists('models_dir_3A89BDdsiwa83Eaeek'):
                print('The temperary folder is missed\n')
                sys.stderr.write('>The temperary folder is missed!\n')
                sys.exit(1)
            for s_A_id in range(predictor.structure_A_num):
                output_model_path = 'models_dir_3A89BDdsiwa83Eaeek/model_A' + str(s_A_id)
                f = open(output_model_path, 'wb')
                pickle.dump(predictor.structure_A_list[s_A_id], open(output_model_path, 'wb'))
                f.close()
            for s_B_id in range(predictor.structure_B_num):
                output_model_path = 'models_dir_3A89BDdsiwa83Eaeek/model_B' + str(s_B_id)
                f = open(output_model_path, 'wb')
                pickle.dump(predictor.structure_B_list[s_B_id], open(output_model_path, 'wb'))
                f.close()
            f_out = open('models_dir_3A89BDdsiwa83Eaeek/flag', 'w')
            f_out.write('deep_ensemble\n'+predictor.predictor_type)
            f_out.close()
            compress_file('model', 'models_dir_3A89BDdsiwa83Eaeek')
            if input_predict==None:
                shutil.rmtree('models_dir_3A89BDdsiwa83Eaeek')
            os.chdir(path_current)
    except BaseException as e:
        print(e)
        print('Saving model failed. Please ensure the path of output model is valid!\n')
        sys.stderr.write('>Saving model failed. Please ensure the path of output model is valid!\n')
        sys.exit(1)

def save_prediction_result(X_pre, Y_pre, headlines, output_folder_path, dic_folder_path):
    try:
        path_current = os.getcwd()
        print(dic_folder_path)
        os.chdir(dic_folder_path)
        dic = read_code.read_dictionary2('models_dir_3A89BDdsiwa83Eaeek/y_dictionary')
        dic_x = read_code.read_dictionary2('models_dir_3A89BDdsiwa83Eaeek/dictionary')
        shutil.rmtree('models_dir_3A89BDdsiwa83Eaeek')
        os.chdir(path_current)
        if output_folder_path[len(output_folder_path)-1]=='/':
            length = len(output_folder_path)
            output_folder_path0 = output_folder_path[:length-1]
        output_result_path = output_folder_path + '/' + 'prediction_result.txt'
        f_out = open(output_result_path, 'w')
        for h in headlines:
            f_out.write(h+'\t')
        f_out.write('predicted_value\n')
        data_len = len(X_pre)
        for i in range(data_len):
            for j in X_pre[i]:
                f_out.write(str(j)+'\t')
            if str(Y_pre[i]) in dic.keys():
                f_out.write(dic[str(Y_pre[i])]+'\n')
            else:
                f_out.write(str(Y_pre[i])+'\n')
        f_out.close()
    except BaseException as e:
        print(e)
        print('Error occured during saving the prediction result. Please check the path of your prediction result.\n')
        sys.stderr.write('>Error occured during saving the prediction result. Please check the path of your prediction result.\n')
        sys.exit(1)

