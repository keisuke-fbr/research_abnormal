#必要ライブラリのインストール
#必要ライブラリのインストール

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.callbacks import EarlyStopping

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


def calcurate_data(train_data, test_data, train_predict_data, test_predict_data):
    #基本データの計算
    errors_train = root_mean_squared_error_per_data(train_data, train_predict_data)
    errors_predict = root_mean_squared_error_per_data(test_data, test_predict_data)

    #トレーニングデータに対する再構成誤差の計算
    reconstract_error = root_mean_squared_error(train_data, train_predict_data)

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

    return abnormal_score, thresold, errors_train_per_features_ave, errors_predict_per_features_ave, thresold_per_features, reconstract_error






#AutoEncoderのモデル
def model_autoencoder(params, init_num, units):
    #ハイパーパラメータの取得
    units_1_3 = units["units_13"]
    units_2 = units["units_2"]
    learning_rate = params["learning_rate"]
    # シード値の抜き出し
    seeds = params["seeds"]

    #２，シード値のセット
    set_seed(seeds[init_num])

    # モデルについて
    #　中間層数は３で設定。中間層１と３は同じユニット数

    # モデルを作成
    model = keras.Sequential(name="autoencoder")


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
    model.add(layers.Dense(units=middle_unit_1, activation="sigmoid" ,input_shape=(input_unit,), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5), seed=seeds[init_num])))
    # 中間層２の作成
    model.add(layers.Dense(units=middle_unit_2, activation="sigmoid", kernel_initializer =initializers.TruncatedNormal(mean=0.0, stddev = pow(units_2, -0.5), seed=seeds[init_num])))
    # 中間層３の作成
    model.add(layers.Dense(units=middle_unit_3, activation="sigmoid", kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5), seed=seeds[init_num])))

    #出力層の作成
    model.add(layers.Dense(units=output_unit, activation="linear", kernel_initializer= initializers.TruncatedNormal(mean=0.0, stddev = pow(175, -0.5), seed=seeds[init_num])))
    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate) 

    model.compile(optimizer=opt,
                        loss=root_mean_squared_error
                       )  

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
    early_stopping_params = {
        "monitor" : "loss",
        "patience" : patience,
        "min_delta" : min_delta,
        'restore_best_weights' : True,
        "mode" : "min"
    }

    # early_stoppingの設定
    early_stopping = EarlyStopping(**early_stopping_params)

    #学習速度高速化のためtf形式に変更する
    # 2. データ型の統一
    train_data = tf.cast(train_data, dtype=tf.float32)  # TensorFlow用に型変換

    history = 0

    #　学習の開始
    history = model.fit(train_data, train_data, epochs = max_epochs, batch_size = batch_size, verbose = detail_log
                        , shuffle = True, validation_data = (train_data,train_data) ,callbacks=[early_stopping] )
    
    # EarlyStoppingで打ち切られたかどうかを確認する
    if len(history.epoch) < max_epochs:
        print(f"モデルは収束しました。訓練は {len(history.epoch)} エポックで停止しました。")
    else:
        print("モデルは最大エポック数まで収束条件に達しませんでした。収束条件を満たさず終了しました。")

    
    return model


#探索ノード数アルゴリズム関数（二分探索法）
#unit_1_3を探索する。探索範囲はminからmaxまで
def next_node(left, right, middle, more_or_less):
    #intは小数点切り捨て
    if more_or_less == "more":
        #大きい方向に進むとき
        left = middle + 1
    
    else:
        #小さい方向に進むとき
        right = middle - 1
    middle = int((right + left)/2)

    return left, right, middle


def search_model(params, train_data, period):
    #最適モデルを格納する変数
    best_model = None

    #unit_2に対するunit1_3の最適ユニット数
    optimal_units = {}
    #各ユニットに対応するmodel
    optimal_models = {}
    #各ユニットに対応する再構成誤差
    reconstract_errors = {}

    #ハイパーパラメータの取得
    max_unit13 = params["max_unit13"]
    min_unit13 = params["min_unit13"]
    max_unit2 = params["max_unit2"]
    error_threshold = params["error_threshold"]

    #シード値のリスト
    seeds = params["seeds"]
    #初期化回数
    num_retry = params["num_retry"]


    #探索開始
    for unit_2 in reversed(range(1,max_unit2+1)):
        #unit_2に対応する最適ユニット数、べストモデル、再構成誤差を記録する変数を作成
        best_units_unit2 = None
        best_model_unit2 = None
        best_reconstract_unit2 = None

        #unit_2に対してunit_13がmax_unit13でも閾値を上回っているかを確認するflagの作成
        flag_stop = False

        #詳細な探索をしたかを確認するflagの作成
        flag_detail_search = False

        #unit2に対するunit1_3の探索
        for unit_1_3 in reversed(range(min_unit13, max_unit13+1, 3)):

            #閾値を下回るかの変数
            flag_low_threshold = False
            
            #初期値の割り振り
            for init_num in range(num_retry):
                #シード値のセット
                set_seed(seeds[init_num])

                #ログの記録
                print("-----------------------------------------------------------------------------------------")
                print(f"探索中: units_1_3={unit_1_3}, units_2={unit_2}, 試行回数={init_num+1}, seed値 = {seeds[init_num]}")
                units = {"units_13":unit_1_3, "units_2":unit_2}

                #モデルの作成
                model = model_autoencoder(params, init_num, units)

                #モデルの学習
                model = learn_model(params, model , train_data)

                #トレーニングデータの再構成誤差の計算
                train_predict = model.predict(train_data, verbose = 0)
                train_errors = root_mean_squared_error(train_data, train_predict)
                print(f"トレーニングデータの再構成誤差：{train_errors}")

                #閾値を下回るかの確認
                if train_errors < error_threshold:
                    print(f"閾値を下回るモデルを発見: units_1_3={unit_1_3}, units_2={unit_2}")
                    print("閾値を下回ったのでノードを減らします")
                    best_units_unit2 = unit_1_3
                    best_model_unit2 = model
                    best_reconstract_unit2 = train_errors
                    flag_low_threshold = True
                    break
                print("閾値を下回りませんでした"+ str(init_num+1) +"回目")

            
            #閾値を下回らず、探索ユニットがmax_unit13である場合、全探索を終了する
            if flag_low_threshold == False and unit_1_3 == max_unit13:
                print(f"unit_2に対してunit_13が{max_unit13}でも閾値を下回らなかったので学習を探索を終了し、次の期間に移ります")
                flag_stop = True
                break

            #最小ユニット数でも閾値を下回った場合、探索を終了する
            if unit_1_3 == min_unit13 and flag_low_threshold == True:
                print(f"unit13の探索が{min_unit13}に達したのでunit2:{unit_2}での探索を終了します")
                print("====================================================================================")
                print(f"最適なノード数が見つかりました: units_1_3={best_units_unit2}, units_2={unit_2}")
                break

            #閾値を下回らなかった場合、ノード数を上げて詳細な探索の開始
            if flag_low_threshold == False:
                flag_detail_search = True
                print(f"閾値を上回ったのでノードを上げて詳細なユニット数を探索します。units_1_3={unit_1_3}, units_2={unit_2}")

                for fine_unit_1_3 in range(unit_1_3+1, unit_1_3+3):

                    flag_low_threshold = False

                    #初期値の割り振り
                    for init_num in range(num_retry):
                        #シード値のセット
                        set_seed(seeds[init_num])

                        #ログの記録
                        print("-----------------------------------------------------------------------------------------")
                        print(f"探索中: units_1_3={fine_unit_1_3}, units_2={unit_2}, 試行回数={init_num+1}, seed値 = {seeds[init_num]}")
                        units = {"units_13":fine_unit_1_3, "units_2":unit_2}

                        #モデルの作成
                        model = model_autoencoder(params, init_num, units)

                        #モデルの学習
                        model = learn_model(params, model , train_data)

                        #トレーニングデータの再構成誤差の計算
                        train_predict = model.predict(train_data, verbose = 0)
                        train_errors = root_mean_squared_error(train_data, train_predict)
                        print(f"トレーニングデータの再構成誤差：{train_errors}")

                        #閾値を下回るかの確認
                        if train_errors < error_threshold:
                            print(f"閾値を下回るモデルを発見: units_1_3={fine_unit_1_3}, units_2={unit_2}")
                            print(f"ノード数を上げてモデルが見つかったのでunit2を更新します")
                            print("====================================================================================")
                            best_units_unit2 = fine_unit_1_3
                            best_model_unit2 = model
                            best_reconstract_unit2 = train_errors
                            flag_low_threshold = True
                            break
                        print("閾値を下回りませんでした"+ str(init_num+1) +"回目")

                    if flag_low_threshold == True:
                        break
                    print(f"units_1_3={fine_unit_1_3}, units_2={unit_2}で閾値を下回りませんでした")

                print(f"最適なノード数が見つかりました: units_1_3={best_units_unit2}, units_2={unit_2}")
            if flag_detail_search == True:
                break
        # flag_stopがtureの場合探索を終了する
        if flag_stop == True:
            break

        #　求まったデータを配列に格納
        optimal_units[unit_2] = best_units_unit2
        optimal_models[(best_units_unit2, unit_2)] = best_model_unit2
        reconstract_errors[(best_units_unit2, unit_2)] = best_reconstract_unit2


    #モデルが見つからなかった場合終わり
    if len(optimal_models) == 0:
        return None

    #求まった最適モデルの組の中で最もノード数が小さいものを求める
    min_total_units = float('inf')
    best_unit_1_3 = None
    best_unit_2 = None
    for unit_2, unit_1_3 in optimal_units.items():
        total_units = unit_1_3*2 + unit_2
    if total_units < min_total_units:
        min_total_units = total_units
        best_unit_1_3 = unit_1_3
        best_unit_2 = unit_2
    print(f"最小ユニットの組み合わせ: units_1_3={best_unit_1_3}, units_2={best_unit_2}, 合計ユニット数={min_total_units},この時の再構成誤差：{reconstract_errors[(best_unit_1_3,best_unit_2)]}")


    #それぞれのユニットの組み合わせと再構成誤差をログに残す
    for unit_2, unit_1_3 in optimal_units.items():
        print(f"各ユニットの組み合わせと対応する再構成誤差の値:unit_2 = {unit_2}, unit_13 = {unit_1_3},この時の再構成誤差:{reconstract_errors[(unit_1_3,unit_2)]}")

    best_model = optimal_models[(best_unit_1_3, best_unit_2)]

    del optimal_units, optimal_models, reconstract_errors

    return best_model




            


