#必要ライブラリのインストール
#必要ライブラリのインストール

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setuptools
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


#モデルの最小化関数
# カスタム損失関数の定義
def custom_loss(delta):
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        small_error_loss = tf.square(error)
        large_error_loss = error
        return tf.where(error < delta, large_error_loss, small_error_loss)
    return loss

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

#異常スコア算出関数
def abnomalScores(originalDF,reduceDF):
  loss = np.sum((np.array(originalDF)-np.array(reduceDF))**2,axis=1)
  loss = pd.Series(data=loss,index=originalDF.index)
  loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
  return loss


#AutoEncoderのモデル
def model_autoencoder(initializer, units_1_3, units_2):
    # モデルについて
    #　中間層数は３で設定。中間層１と３は同じユニット数

    # モデルを作成
    model = keras.Sequential(name="autoencoder")

    #重みの初期化方法
    initializer = initializer

    #入力層の情報
    input_unit = 11

    #中間層の情報
    middle_unit_1 = units_1_3
    middle_unit_2 = units_2
    middle_unit_3 = units_1_3
   

    #出力層の情報
    output_unit = 11

    #層の追加
    #中間層１の作成
    model.add(layers.Dense(units=middle_unit_1, activation="relu" ,input_shape=(input_unit,), kernel_initializer=initializer))
    # 中間層２の作成
    model.add(layers.Dense(units=middle_unit_2, activation="relu", kernel_initializer=initializer))
    # 中間層３の作成
    model.add(layers.Dense(units=middle_unit_3, activation="relu", kernel_initializer=initializer))

    #出力層の作成
    model.add(layers.Dense(units=output_unit, activation="sigmoid", kernel_initializer=initializer))
    #モデルのコンパイル
    #model.compile(optimizer="adam", loss=custom_loss(delta=0.5))
    model.compile(optimizer="adam", loss=root_mean_squared_error)

    return model


#モデル探索の学習関数、モデルと学習データを入力することで学習を完了させ、その損失が得られる
#学習アルゴリズムはmax_epochs回学習させ、損失関数に変化がなくなった時点で学習を終了させる
def train_and_evaluate(model, X_train, max_epochs, early_stopping):
    history = model.fit(X_train, X_train, epochs=max_epochs, batch_size=32, verbose=0,
                        validation_split=0.2, callbacks=[early_stopping])
    final_loss = history.history['val_loss'][-1]  # 最終エポックの検証損失

    # 学習データに対する再構成誤差を計算
    reconstructed = model.predict(X_train)
    reconstruction_errors = np.mean(np.square(X_train - reconstructed), axis=1)  # 各サンプルの再構成誤差
    
    # 最も再構成誤差が大きかったデータポイントの誤差（異常スコアの閾値）
    max_error = np.max(reconstruction_errors)
    
    print(f"最も大きな再構成誤差（異常スコアの閾値）: {max_error}")
    
    return final_loss, max_error


def search_model(initializer, X_train, error_threshold, max_epochs, early_stopping, num_initializations=4):
    best_model = None
    best_units = None
    best_loss = float('inf')
    best_threshold = None  # 異常スコアの閾値
    
    # 探索開始
#    for units_1_3 in range(2, 11):
#        for units_2 in range(1, units_1_3):
    for units_1_3 in reversed(range(2, 30)):
        for units_2 in reversed(range(1, 11)):
            #初期点を何回か振り分けて収束を考える
            for init_num in range(num_initializations):  # 初期化方法を複数回試行
                initializer = initializer  # 初期化方法の設定
                print("----------------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------------")
                print(f"探索中: units_1_3={units_1_3}, units_2={units_2}, 初期化方法={initializer}, 試行回数={init_num+1}")
                
                model = model_autoencoder(initializer, units_1_3, units_2)
                print("中間ユニット(1・3):"+str(units_1_3)+ "個,中間ユニット(2):"+ str(units_2)+ "個、初期化回数" + str(init_num+1) +"回目のモデル構築完了")
                final_loss, max_error = train_and_evaluate(model, X_train, max_epochs, early_stopping)
                print("中間ユニット(1・3):"+str(units_1_3)+ "個,中間ユニット(2):"+ str(units_2)+ "個、初期化回数" + str(init_num+1) +"回目のモデル学習")
                print(f"最終損失: {final_loss}")

                print("------------------------------------------------------------------------------------------")
                
                if final_loss < best_loss:  # 最も良い損失を持つモデルを保存
                    best_model = model
                    best_units = (units_1_3, units_2)
                    best_loss = final_loss
                    best_threshold = max_error  # 異常スコアの閾値を保存
                    print(f"新しい最適モデル発見: units_1_3={units_1_3}, units_2={units_2}, 初期化方法={initializer}, 最終損失={final_loss}")
                    
                if final_loss < error_threshold:  # 閾値を下回る場合探索を終了
                    print("====================================================================================")
                    print("====================================================================================")
                    print(f"閾値を下回るモデルを発見: units_1_3={units_1_3}, units_2={units_2}")
                    print(f"異常スコアの閾値: {best_threshold}")
                    print("====================================================================================")
                    print("====================================================================================")
                    return best_model ,best_threshold  # 最適なモデルと異常スコアの閾値を返却

    if best_model:
        print(f"最適なモデルを発見: units_1_3={best_units[0]}, units_2={best_units[1]}, 最終損失={best_loss}")
        print(f"異常スコアの閾値: {best_threshold}")
        return best_model, best_threshold
    else:
        print("閾値を下回るモデルは見つかりませんでした。")
        return None,None


#15期間においてモデルを作成し、結果を格納する関数
#各期間においてほしい情報は、日付に対応する異常スコア、閾値
def result(data_ex, colums_list,  initializer,error_threshold, max_epochs, early_stopping, num = 15  ):

    #結果を格納する配列
    results_df = pd.DataFrame(columns=["measurement_date", "anomaly_score"])
    #各期間における閾値を格納する配列
    thresholds = []


    #初期データの抜き出し及び、繰り返し処理の準備
    data_trainstart = data_ex["measurement_date"][0]
    data_trainstart_year = data_trainstart.year
    data_trainstart_month = data_trainstart.month
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
        best_model, best_threshold = search_model(initializer,train_data, error_threshold, max_epochs, early_stopping )
        print("====================================================================================================")
        print(str(i+1) + "回目の期間のモデル作成終了")
        print("====================================================================================================")
        #結果
        result_data = best_model.predict(test_data)
    
        #pandas形式へ変換
        test_data = pd.DataFrame(test_data,columns=colums_list)
        result_data = pd.DataFrame(result_data,columns=colums_list)
    
        #異常値の算出
        abnormal_value = abnomalScores(test_data,result_data)
    
    
        #結果の格納
        temp_df = pd.DataFrame({
            "measurement_date": test_data_origin["measurement_date"].values,
            "anomaly_score": abnormal_value
        })
    
        #結果の結合
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
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
    
    return  results_df, thresholds
        
        
    


#結果を算出する関数
def experiment(data_ex, colums_list, initializer,error_threshold, max_epochs, early_stopping):

    #初期データの抜き出し及び、繰り返し処理の準備
    data_trainstart = data_ex["measurement_date"][0]
    data_trainstart_year = data_trainstart.year
    data_trainstart_month = data_trainstart.month
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
    
   

    #15期間における実験の実行
    results_df, thresholds = result(data_ex, colums_list, data_trainend, data_trainstart, data_testend, data_teststart, initializer,error_threshold, max_epochs, early_stopping)

    return results_df, thresholds