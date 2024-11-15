#実験が完了したときに各特徴量や閾値の学習がどのようになっているかを確認する

#必要ライブラリのインストール

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from .. import model_basic

def search_model(initializer, X_train, error_threshold, max_epochs, early_stopping_params, units, loss_array ,period, num_initializations=4):
    best_model = None
    best_units = None
    best_loss = float('inf')

    # シード値を準備する
    seed_dict = {}
    for units_1_3 in range(12, 31):
        for units_2 in range(2, 11):
            base_seed = 40 + (units_1_3 - 12) * 9 + (units_2 - 2)  # 基準となるシード値の生成
            seed_dict[(units_1_3, units_2)] = [base_seed + i for i in range(4)]  # 各組み合わせに対して4つのシード値

    # 各実験に対するノード数の定義
    ex_node = {
        "decrease":[[29,10],[15,10],[29,10],[12,10]],
        "skip_decrease":[[12,10],[12,10],[12,10],[12,10]],
        "binary":[[12,10],[30,10],[12,10],[28,10]]
    }

    #各実験におけるベストモデルを格納する配列の準備
    result_per_ex = {}



    # 実験開始
    #　各手法ごとに実験を行う
    for key, value in ex_node.items():
        print("===============================================================================================================")
        print(f"実験：{key}")

        units_1_3 = value[period][0]
        units_2 = value[period][1]

        # 現在の組み合わせに対応するシードリストを取得
        seeds = seed_dict[(12, 2)]
        best_model = None

        best_loss = float("inf")

        for init_num in range(num_initializations):

            model_basic.set_seed(seeds[init_num])
            initializer = initializer

            print(f"実験：{key}の{init_num+1}回目の実験：ノード数はunit1_3:{units_1_3},unit_2{units_2}")
            print(f"seed値：{seeds[init_num]}")

            model = model_basic.model_autoencoder(initializer, units_1_3, units_2)
            final_loss = model_basic.train_and_evaluate(model, X_train, max_epochs, early_stopping_params)

            if final_loss < best_loss:
                print(f"modelの更新：final_loss:{final_loss}")
                best_model = model
                best_loss = final_loss

        result_per_ex[key] = best_model

    return result_per_ex, loss_array




#　各実験ごとのモデルから特徴量ごとの予測再構成誤差と、トレーニング再構成誤差を返す関数
def analysis_error_per_features(result_per_ex, error_per_traindata, error_per_predictdata,abnormal_values ,train_data, test_data, colums_list,period):
    
    error_per_traindata[period] = {}
    error_per_predictdata[period] = {}
    abnormal_values[period] = {}

    #各手法ごとに処理を行う
    for key, value in result_per_ex.items():
        model = value

        #テストデータの予測値
        result_data = model.predict(test_data)
        #トレーニングデータの再構成値
        traindata_model = model.predict(train_data)

        #再構築データ数
        print("再構築データ数 : " + str(len(traindata_model)))
    
        #pandas形式へ変換
        train_data = pd.DataFrame(train_data, columns=colums_list)
        print(f"train_dataの構造：{train_data.shape}")

        test_data = pd.DataFrame(test_data,columns=colums_list)
        print(f"test_dataの構造：{test_data.shape}")

        result_data = pd.DataFrame(result_data,columns=colums_list)
        print(f"result_dataの構造：{result_data.shape}")

        traindata_model = pd.DataFrame(traindata_model, columns=colums_list)
        print(f"traindata_modelの構造：{traindata_model.shape}")

        #  print(f"手法{key}のモデル結果：{result_data}")

    
        #異常値の算出
        abnormal_value = model_basic.biggest_threshold(test_data,result_data)
        print(f"abnormal_value:{abnormal_value}")

        #　各特徴量における再構成誤差を算出する
        errors_predict = model_basic.reconstruction_error_per_feature(test_data,result_data)
        print(f"errors_predict:{errors_predict}")

        # 閾値として選ばれたデータの各特徴量ごとの値を算出する
        errors_train = model_basic.biggest_threshold_per_features(train_data, traindata_model)
        print(f"errors_train:{errors_train}")

        error_per_traindata[period][key] = errors_train

        error_per_predictdata[period][key] = errors_predict

        abnormal_values[period][key] = abnormal_value

    return error_per_predictdata, error_per_traindata, abnormal_values




#15期間においてモデルを作成し、結果を格納する関数
#各期間においてほしい情報は、日付に対応する異常スコア、閾値
def result(data_ex, colums_list,  initializer,error_threshold, max_epochs, early_stopping_params, units, num = 4  ):

    #異常スコアを格納する配列
    results_df = pd.DataFrame(columns=["measurement_date"] + ["anomaly_score"])

    #トレーニングデータの再構成値を格納する配列
    traindata_model_df = pd.DataFrame(columns=["measurement_date"] + colums_list)
    

    #lossの値を格納する配列を作成
    units2_range = range(1, 11)       # 1から10まで
    units1_3_range = range(12, 31)    # 12から30まで
    # 三次元配列を辞書で表現
    loss_array = {
        units2: {units1_3: [] for units1_3 in units1_3_range} for units2 in units2_range
    }



    #初期データの抜き出し及び、繰り返し処理の準備
    data_trainstart = data_ex["measurement_date"][0]

    data_trainstart_year = data_trainstart.year
    data_trainstart_month = data_trainstart.month
    data_trainstart = str(data_trainstart_year) + "/" + str(data_trainstart_month) + "/01 00:00:00"

    #手動でデータの設定
    data_trainstart = "2016/6/01 00:00:00"
    data_trainstart_year = 2016
    data_trainstart_month = 6

    data_trainend_year = data_trainstart_year + 2
    data_trainend_month = data_trainstart_month
    data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"
    
    #手動でデータの設定
    data_trainend = "2018/6/01 00:00:00"
    data_trainend_year = 2018
    data_trainend_month = 6
    
    data_teststart = data_trainend
    if data_trainend_month == 12:
        data_testend_year = data_trainend_year + 1
        data_testend_month = 1
    else:
        data_testend_year = data_trainend_year
        data_testend_month = data_trainend_month + 1
    
    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

    #　各期間における実験ごとの特徴量のデータを保持する配列の作成
    #　トレーニングデータに対して
    errors_per_traindata = {}
    # 予測値に対して
    errors_per_predictdata = {}
    #　閾値の値
    abnormal_values = {}

    #モデル実装部分
    for i in range(num):
        print("=====================================================================================")
        print(str(i+1) + "回目の期間のモデル")

        #i回目のデータの抜き出し
        #トレーニングデータの抜き出し
        train_data = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        train_data = train_data[colums_list]
        train_data = train_data.values
    
    
        #テストデータの抜き出し
        #テストデータはdata_trainendの一か月である
        test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        test_data_origin = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        test_data = test_data[colums_list]
        test_data = test_data.values


        #該当データ期間
        print("該当期間の確認")
        print("はじまり(train)" + str(data_trainstart))
        print("終わり(train)" + str(data_trainend))
        print("はじまり(test)" + str(data_teststart))
        print("終わり(test)" + str(data_testend))
    

        #初期点を変更しながら、最適モデルを構築する
        print(str(i+1) + "回目の期間のモデル作成開始")
        print(str(i+1) + "回目の期間のトレーニングデータ数:" + str(len(train_data)))

        #モデルの作成
        #各実験ごとのbestmodelが返される
        results_per_ex, loss_array = search_model(initializer,train_data, error_threshold, max_epochs, early_stopping_params, units, loss_array, i)

        print("====================================================================================================")
        print(str(i+1) + "回目の期間のモデル作成終了")
        print("====================================================================================================")

        errors_train, errors_predict, abnormal_values = analysis_error_per_features(results_per_ex, errors_per_traindata, errors_per_predictdata,abnormal_values , train_data, test_data, colums_list, i)


        #結果
        print(str(i+1) + "回目の期間のテストデータ数:" + str(len(test_data)))



        print("traindata_model_dfの日付の最大値:" + str(traindata_model_df["measurement_date"].max()))
        print("trainの日付の最大値:" + str(data_trainend)) 
        print("testの日付の最大値:" + str(data_testend))


        #日付の更新
        if data_trainstart_month == 12:
            data_trainstart_year += 1
            data_trainstart_month = 1
        else:
            data_trainstart_month += 1
        data_trainstart = str(data_trainstart_year) + "/" + str(data_trainstart_month) + "/01 00:00:00"        
        
        data_trainend_year = data_trainstart_year + 2
        data_trainend_month = data_trainstart_month
        data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"
    
        data_teststart = data_trainend
        if data_trainend_month == 12:
            data_testend_year = data_trainend_year + 1
            data_testend_month = 1
        else:
            data_testend_year = data_trainend_year
            data_testend_month = data_trainend_month + 1
    
        data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"


    #データの確認ログ
    print("data_exの日付の最大値:" + str(data_ex["measurement_date"].max()))
    print("traindata_model_dfの日付の最大値:" + str(traindata_model_df["measurement_date"].max()))
    print("data_exのデータ数:" + str(len(data_ex)))
    print("traindata_model_dfのデータ数:" +str(len(traindata_model_df)))
    # 日付が文字列で保存されている場合、日付型に変換します
    data_ex['measurement_date'] = pd.to_datetime(data_ex['measurement_date'])
    # 2019年2月のデータをフィルタリングして、そのデータ数をカウントします
    data_feb_2019 = data_ex[(data_ex['measurement_date'].dt.year == 2019) & (data_ex['measurement_date'].dt.month == 2)]
    # データ数を表示
    feb_2019_count = data_feb_2019.shape[0]
    print(f"data_exの最終月に含まれるデータ数:{feb_2019_count}")

    return  errors_train, errors_predict, abnormal_values
        
        
    
    