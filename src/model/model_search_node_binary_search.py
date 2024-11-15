#必要ライブラリのインストール

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setuptools
import random
import tensorflow as tf
from tensorflow import keras


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from .. import model_basic


def search_model(initializer, X_train, error_threshold, max_epochs, early_stopping_params, units, loss_array, num_initializations=4):
    max_unit_1_3 = 30
    min_unit_1_3 = units + 1
    best_model = None


    # シード値を準備する
    seed_dict = {}
    for units_1_3 in range(12, 31):
        for units_2 in range(2, 11):
            base_seed = 40 + (units_1_3 - 12) * 9 + (units_2 - 2)  # 基準となるシード値の生成
            seed_dict[(units_1_3, units_2)] = [base_seed + i for i in range(4)]  # 各組み合わせに対して4つのシード値


    #探索開始
    #unit_2に対するunit1_3の最適ユニット数
    optimal_units = {}
    #各ユニットに対応するmodel
    optimal_models = {}
    
    for units_2 in reversed(range(1,11)):
        flag_stop = 0
        optimal_unit = 0
        more_or_less = 0
        left = min_unit_1_3
        right = max_unit_1_3
        units_1_3, left, right = model_basic.next_node_num(left, right, 0, more_or_less)

        best_model_unit_2 = None

        print(f"新しくモデルを探索します：units_2={units_2}")
        #探索した数値を格納するリスト
        search_list = []    
       
        while(True):

            # 現在の組み合わせに対応するシードリストを取得
            seeds = seed_dict[(units_1_3, units_2)]

            for init_num in range(num_initializations):
                initializer = initializer  # 初期化方法の設定
                 # 各初期化ごとに対応するシードを設定
                set_seed(seeds[init_num])

                print("----------------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------------")
                print(f"探索中: units_1_3={units_1_3}, units_2={units_2}, 初期化方法={initializer}, 試行回数={init_num+1}")
                print(f"seed値:{seeds[init_num]}")
                model = model_basic.model_autoencoder(initializer, units_1_3, units_2)

                print("中間ユニット(1・3):"+str(units_1_3)+ "個,中間ユニット(2):"+ str(units_2)+ "個、初期化回数" + str(init_num+1) +"回目のモデル構築完了")
                final_loss = model_basic.train_and_evaluate(model, X_train, max_epochs, early_stopping_params)
                loss_array[units_2][units_1_3].append(final_loss)
                print("中間ユニット(1・3):"+str(units_1_3)+ "個,中間ユニット(2):"+ str(units_2)+ "個、初期化回数" + str(init_num+1) +"回目のモデル学習")
                print(f"最終損失: {final_loss}")
                print("------------------------------------------------------------------------------------------")
                
                #閾値を下回ったらその時点でOK
                if final_loss < error_threshold:
                    print("====================================================================================")
                    print(f"閾値を下回るモデルを発見: units_1_3={units_1_3}, units_2={units_2}")
                    print("閾値を下回ったのでノード数を減らします")
                    print("====================================================================================")
                    #最良モデルの更新
                    more_or_less = 2
                    best_model_unit_2 = model
                    optimal_unit = units_1_3
                    break
            
            if final_loss >= error_threshold:
                print("====================================================================================")
                print(f"閾値を下回りませんでした: units_1_3={units_1_3}, units_2={units_2}")
                print("閾値を下回らなかったのでノード数を増やします")
                print("====================================================================================")
                more_or_less = 1

            #最適なノード数が見つからなく終了
            #ノード数を増やしても無理だった
            if more_or_less == 1 and units_1_3 == max_unit_1_3:
                print("====================================================================================")
                print(f"最適なノード数が見つかりませんでした: units_1_3={units_1_3}, units_2={units_2}")
                print("====================================================================================")
                optimal_models[(10000,units_2)] = model
                optimal_units[units_2] = 10000
                flag_stop = 5
                break


            #ノード数を減らしても行けた
            if units_1_3 == 12 and more_or_less == 2:
                print("====================================================================================")
                print(f"unit12でも十分でした: units_1_3={units_1_3}, units_2={units_2}")
                print("====================================================================================")
                optimal_unit = units_1_3
                optimal_models[(optimal_unit,units_2)] = best_model_unit_2
                optimal_units[units_2] = optimal_unit
                break

                
            #ノード数と範囲の更新
            print(f"元の探索範囲 : right={right}, left = {left}")
            units_1_3, left, right = model_basic.next_node_num(left ,right, units_1_3, more_or_less) 
            print(f"新たな探索範囲 : right={right}, left = {left}")
            print("ノード数の更新"+ str(units_1_3) + "へ")
            

            

            if units_1_3 in search_list:
                print("すでに探索したノード数なので探索を終了します")
                print("====================================================================================")
                print(f"最適なノード数が見つかりました: units_1_3={optimal_unit}, units_2={units_2}")
                print("====================================================================================")
                optimal_models[(optimal_unit,units_2)] = best_model_unit_2
                optimal_units[units_2] = optimal_unit

                break

            search_list.append(units_1_3)
            
        #どんなにunit1_3の表現力を上げても十分なモデルを作成できない、またはモデルの探索が最後まで行ったら終わり   
        if flag_stop == 5 or units_2 == 1:
            break
        
    #今、各unit_1_3の組み合わせにおいて最適なunit2の値が対応している。その中で最もノード数が小さいものをベストモデルとする。
    # unit_2とunit_1_3の合計が最小の組み合わせを見つける
    min_total_units = float('inf')
    best_unit_1_3 = None
    best_unit_2 = None

    for unit_2, unit_1_3 in optimal_units.items():
        total_units = unit_1_3*2 + unit_2
    if total_units < min_total_units:
        min_total_units = total_units
        best_unit_1_3 = unit_1_3
        best_unit_2 = unit_2

    print(f"最小ユニットの組み合わせ: units_1_3={best_unit_1_3}, units_2={best_unit_2}, 合計ユニット数={min_total_units}")
    best_model = optimal_models[(best_unit_1_3, best_unit_2)]

    del optimal_units, optimal_models

    return best_model, loss_array

#15期間においてモデルを作成し、結果を格納する関数
#各期間においてほしい情報は、日付に対応する異常スコア、閾値
def result(data_ex, colums_list,  initializer,error_threshold, max_epochs, early_stopping_params, units, num = 4  ):

    #異常スコアを格納する配列
    results_df = pd.DataFrame(columns=["measurement_date"] + ["anomaly_score"])

    #トレーニングデータの再構成値を格納する配列
    traindata_model_df = pd.DataFrame(columns=["measurement_date"] + colums_list)
    
    #各期間における閾値を格納する配列
    thresholds = []

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
        best_model, loss_array = search_model(initializer,train_data, error_threshold, max_epochs, early_stopping_params, units, loss_array)

        print("====================================================================================================")
        print(str(i+1) + "回目の期間のモデル作成終了")
        print("====================================================================================================")


        #結果
        print(str(i+1) + "回目の期間のテストデータ数:" + str(len(test_data)))

        #テストデータの予測値
        result_data = best_model.predict(test_data)

        #トレーニングデータの再構成値
        traindata_model = best_model.predict(train_data)

        #再構築データ数
        print("再構築データ数 : " + str(len(traindata_model)))
    
        #pandas形式へ変換
        test_data = pd.DataFrame(test_data,columns=colums_list)
        result_data = pd.DataFrame(result_data,columns=colums_list)
        traindata_model = pd.DataFrame(traindata_model, columns=colums_list)
    
        #異常値の算出
        abnormal_value = model_basic.abnomalScores(test_data,result_data)

        #最も再構成誤差の大きい特徴の抽出
        best_threshold = model_basic.biggest_threshold(train_data, traindata_model)
        print(f"異常スコアの閾値: {best_threshold}")
    
    
        #今期間の異常スコア、予測の結果の格納
        temp_df = pd.DataFrame({
            "measurement_date": test_data_origin["measurement_date"].values,
            "anomaly_score": abnormal_value
        })


        train_data_date = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        temp_traindata_df = pd.DataFrame({
            "measurement_date" : train_data_date["measurement_date"].values
        })

        #再構成値の格納
        for col in colums_list:
            temp_traindata_df[col] = traindata_model[col]

        
    
        #これまでの期間の結果の結合
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        traindata_model_df = pd.concat([traindata_model_df, temp_traindata_df],ignore_index=True)


        print("traindata_model_dfの日付の最大値:" + str(traindata_model_df["measurement_date"].max()))
        print("trainの日付の最大値:" + str(data_trainend)) 
        print("testの日付の最大値:" + str(data_testend))

        #各期間の閾値を格納
        thresholds.append({
            "term" : i+1,
            "threshold" : best_threshold,
            "test_start" : data_teststart,
            "test_end" : data_testend
        })

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

    #再構築データに関して、期間が被る部分があるので同じ日付を予測している箇所がある。よって平均をとってデータ数をそろえる。
    traindata_model_df = traindata_model_df.groupby("measurement_date").mean().reset_index()

    #各ノードに対して平均をとる
    loss_array = {
        units2: {
            units1_3: (sum(values) / len(values)) if len(values) > 0 else None
            for units1_3, values in units1_3_dict.items()
        }   
        for units2, units1_3_dict in loss_array.items()
    }
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
    return  results_df, traindata_model_df, thresholds, loss_array
        
        
    