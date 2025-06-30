#必要ライブラリのインストール
#必要ライブラリのインストール

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.callbacks import EarlyStopping

from custom_class.custom_tensorflow import CustomAutoencoder
from custom_class.custom_tensorflow import MaxReconstructionErrorEarlyStopping

import config



# シード値を決定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#データの再構成値を計算する関数
def root_mean_squared_error(y_true, y_pred):
    errors = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))
    return errors

#各データに対して再構成値を計算する関数
def root_mean_squared_error_per_data(y_true, y_pred):
    errors =tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true),axis=1))
    #データ行１列に変換
    errors = tf.reshape(errors,(-1,1))
    return errors


#各特徴量ごとの再構成誤差を計算する関数
def calcurate_reconstraction_error_per_features(y_true, y_pred):
    #差を取って二乗する計算
    reconstraction_errors = np.sqrt((y_true - y_pred) ** 2)

    #各特徴量ごとに平均をとる 
    mean_errors_per_feature = np.mean(reconstraction_errors,axis=0)
    
    return mean_errors_per_feature



def calcurate_data(train_data, test_data, train_predict_data, test_predict_data, flag_skip=0):
    #基本データの計算
    errors_train = root_mean_squared_error_per_data(train_data, train_predict_data)
    errors_predict = root_mean_squared_error_per_data(test_data, test_predict_data)

    #トレーニングデータに対する再構成誤差の計算
    reconstract_error = root_mean_squared_error(train_data, train_predict_data)

    #各データごとの再構成誤差
    errors_train_per_data = errors_train

    errors_train_per_features = calcurate_reconstraction_error_per_features(train_data,train_predict_data)
    errors_train_per_features = pd.DataFrame([errors_train_per_features], columns=config.columns_list)

    errors_predict_per_features = calcurate_reconstraction_error_per_features(test_data,test_predict_data)
    errors_predict_per_features = pd.DataFrame([errors_predict_per_features], columns=config.columns_list)

    #各特徴量、各データのtrain_dataの再構成誤差
    errors_per_data_per_features = np.sqrt((train_data - train_predict_data) ** 2)
    errors_per_data_per_features = pd.DataFrame(errors_per_data_per_features,columns=config.columns_list)

    #テストデータの異常度の計算
    abnormal_score = errors_predict
    #一次元配列に変更
    abnormal_score = abnormal_score.numpy().flatten()

    #閾値の計算
    thresold = errors_train.numpy().max()

    #特徴量ごとのトレーニングデータに対する再構成誤差の平均
    errors_train_per_features = pd.DataFrame(errors_train_per_features, columns= config.columns_list)
    errors_train_per_features_ave = errors_train_per_features[config.columns_list].mean()

    #特徴量ごとのテストデータに対する再構成誤差の平均
    errors_predict_per_features = pd.DataFrame(errors_predict_per_features, columns=config.columns_list)
    errors_predict_per_features_ave = errors_predict_per_features[config.columns_list].mean()

    #閾値データの特徴量ごとの再構成誤差
    max_position = int(tf.argmax(errors_train).numpy())
    print(f"最も再構成誤差が大きかったインデックス番号：{max_position}")
    thresold_per_features = errors_per_data_per_features.iloc[max_position]

    

    if flag_skip == 0:
        return abnormal_score, thresold, errors_train_per_features_ave, errors_predict_per_features_ave, thresold_per_features, reconstract_error, errors_train_per_data
    
    else:
        return abnormal_score, errors_predict_per_features_ave
