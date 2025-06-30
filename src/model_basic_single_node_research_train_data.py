#必要ライブラリのインストール
#必要ライブラリのインストール

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.callbacks import EarlyStopping

from custom_class.custom_learning_average_stopping_maxdata_max_stop import CustomAutoencoder
from custom_class.custom_learning_average_stopping_maxdata_max_stop import DualConditionEarlyStopping

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






#AutoEncoderのモデル
def model_autoencoder(params, init_num, period):
    #ハイパーパラメータの取得
    unit13 = params["units"][period][0]
    unit2 = params["units"][period][1]
    learning_rate = params["learning_rate"]
    # シード値の抜き出し
    seeds = params["seeds"]

    #２，シード値のセット
    set_seed(seeds[init_num])

    # モデルについて
    #　中間層数は３で設定。中間層１と３は同じユニット数

    # モデルを作成
    model = keras.Sequential(name="autoencoder")


    # ユニット数の定義
    input_unit = 11
    middle_unit_1 = unit13
    middle_unit_2 = unit2
    middle_unit_3 = unit13
    output_unit = 11

    # モデル構造の定義（Functional API）
    inputs = keras.Input(shape=(input_unit,))
    x = layers.Dense(units=middle_unit_1, activation="sigmoid",
                     kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=pow(unit13, -0.5), seed=seeds[init_num]))(inputs)
    x = layers.Dense(units=middle_unit_2, activation="sigmoid",
                     kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=pow(unit2, -0.5), seed=seeds[init_num]))(x)
    x = layers.Dense(units=middle_unit_3, activation="sigmoid",
                     kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=pow(unit13, -0.5), seed=seeds[init_num]))(x)
    outputs = layers.Dense(units=output_unit, activation="linear",
                           kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=pow(175, -0.5), seed=seeds[init_num]))(x)

    model = CustomAutoencoder(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error)

    return model



# 実際に学習を行う関数
#@tf.function
def learn_model(params, model , train_data):
    #１，ハイパーパラメータをばらす
    batch_size = params["batch_size"]
    patience = params["patience"]
    min_delta = params["min_delta"]
    max_epochs = params["max_epochs"]
    detail_log = params["detail_log"]
    patience_max = params["patience_max"]
    min_delta_max = params["min_delta_max"]
    confirm_datapoint = params["confirm_datapoint"]

    early_stopping_params = {
        "monitor" : "loss",
        "patience" : patience,
        "min_delta" : min_delta,
        "patience_max" : patience_max,
        "min_delta_max" : min_delta_max,
        'restore_best_weights' : True,
        "detail_log" : detail_log,
        "mode" : "min",
        "confirm_datapoint" : confirm_datapoint
    }

    # early_stoppingの設定
    early_stopping = DualConditionEarlyStopping(model, early_stopping_params)

    # 訓練データをモデルに渡す
    model.train_data_for_monitoring = train_data

    #学習速度高速化のためtf形式に変更する
    # 2. データ型の統一
    train_data = tf.cast(train_data, dtype=tf.float32)  # TensorFlow用に型変換

    history = 0

    #　学習の開始
    history = model.fit(train_data, train_data, epochs = max_epochs, batch_size = batch_size, verbose = 0
                        , shuffle = True, validation_data = (train_data,train_data) ,callbacks=[early_stopping] )
    
    # EarlyStoppingで打ち切られたかどうかを確認する
    if len(history.epoch) < max_epochs:
        print(f"モデルは収束しました。訓練は {len(history.epoch)} エポックで停止しました。")
    else:
        print("モデルは最大エポック数まで収束条件に達しませんでした。収束条件を満たさず終了しました。")

    
    return model



def single_model(params, train_data ,period):

    #unit_2に対するunit1_3の最適ユニット数
    optimal_units = {}
    #各ユニットに対応するmodel
    optimal_models = {}
    #各ユニットに対応する再構成誤差
    reconstract_errors = {}

    #ハイパーパラメータの取得
    unit13 = params["units"][period][0]
    unit2 = params["units"][period][1]
    now_errors = 10000

    #シード値のリスト
    seeds = params["seeds"]
    #初期化回数
    num_retry = params["num_retry"]




            
    #初期値の割り振り
    for init_num in range(num_retry):
        #シード値のセット
        set_seed(seeds[init_num])

        #ログの記録
        print("-----------------------------------------------------------------------------------------")
        print(f"探索中: units_1_3={unit13}, units_2={unit2}, 試行回数={init_num+1}, seed値 = {seeds[init_num]}")
        #モデルの作成
        model = model_autoencoder(params, init_num, period)
        #モデルの学習
        model = learn_model(params, model , train_data)

        #値のセット
        model.train_data_for_monitoring = train_data

        #トレーニングデータの再構成誤差の計算
        train_predict = model.predict(train_data, verbose = 0)
        train_errors = root_mean_squared_error(train_data, train_predict)
        train_errors_max = root_mean_squared_error_per_data(train_data, train_predict).numpy().max()
        print(f"トレーニングデータの再構成誤差（平均）：{train_errors}")
        print(f"トレーニングデータの再構成誤差（最大値）：{train_errors_max}")
        #閾値を下回るかの確認
        if train_errors < now_errors:
            print(f"モデルの更新: シード値={seeds[init_num]}")
            best_model_unit2 = model
            best_reconstract_unit2 = train_errors
            now_errors = train_errors

    print(f"結果: units_1_3={unit13}, units_2={unit2},この時の再構成誤差：{best_reconstract_unit2}")

    del optimal_units, optimal_models, reconstract_errors

    return best_model_unit2




            


