#必要ライブラリのインストール
#必要ライブラリのインストール

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.callbacks import EarlyStopping

import os
import sys


from custom_Class.custom_earlystopping import CustomAutoencoder
from custom_Class.custom_earlystopping import MaxReconstructionErrorEarlyStopping

from keras.saving import register_keras_serializable

import multiprocessing
from functools import partial
import logging
import config



# シード値を決定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#データの再構成値を計算する関数
@register_keras_serializable()
def root_mean_squared_error(y_true, y_pred):
    mse_total = tf.reduce_mean(tf.square(y_pred - y_true))  # スカラー
    rmse_total = tf.sqrt(mse_total)  # スカラー → RMSE_total)
    return rmse_total

#各データに対して再構成値を計算する関数
def root_mean_squared_error_per_data(y_true, y_pred):
    mse_per_sample = tf.reduce_mean(tf.square(y_pred - y_true), axis=1)
    rmse_per_sample = tf.sqrt(mse_per_sample)
    #データ行１列に変換
    errors = tf.reshape(rmse_per_sample,(-1,1))
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
def model_autoencoder(params, init_num, unit13, unit2):
    #ハイパーパラメータの取得

    learning_rate = params["learning_rate"]
    # シード値の抜き出し
    seeds = params["seeds"]

    #２，シード値のセット
    set_seed(seeds[init_num])

    # モデルについて
    #　中間層数は３で設定。中間層１と３は同じユニット数

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
def learn_model(params, model , train_data, logger):
    #１，ハイパーパラメータをばらす
    batch_size = params["batch_size"]
    max_epochs = params["max_epochs"]
    early_stopping_params = params["early_stopping_params"]


    # early_stoppingの設定
    early_stopping = MaxReconstructionErrorEarlyStopping(model, early_stopping_params, logger=logger)

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
    if hasattr(model, 'reached_threshold') and model.reached_threshold:
        logger.info(f"モデルは EarlyStopping により閾値を満たして停止しました。（{len(history.epoch)} エポック）")
    else:
        logger.info("モデルは最大エポック数まで収束条件に達しませんでした。収束条件を満たさず終了しました。")

    
    return model


def initialize_log_files(seeds,period_log):
    for seed in seeds:
        log_filename = f"log_seed_{period_log}_{seed}.log"
        with open(log_filename, mode='w', encoding='utf-8') as f:
            f.write('')  # 空にする


def set_up_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # 既存の全ての handler を削除（重要！！）
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


#　分散処理のための関数,各初期値での実行
def run_one_seed(params, train_data, unit_1_3, unit_2, period_log, init_num):

    # seedの取り出し
    seed = params["seeds"][init_num]  # 例：seeds = [50, 51, 52, 53]
    set_seed(seed)


    log_filename = f"log_seed_{period_log}_{seed}.log"
    logger = set_up_logger(log_filename)

    logger.info("-----------------------------------------------------------------------------------------")
    logger.info(f"探索中: units_1_3={unit_1_3}, units_2={unit_2}, 試行回数={init_num+1}, seed値 = {seed}")

    model = model_autoencoder(params, init_num, unit_1_3, unit_2)
    model = learn_model(params, model, train_data, logger)


    #トレーニングデータの再構成誤差の計算
    train_predict = model.predict(train_data, verbose = 0)
    train_errors = root_mean_squared_error(train_data, train_predict)
    train_errors_max = root_mean_squared_error_per_data(train_data, train_predict).numpy().max()

    logger.info(f"トレーニングデータの再構成誤差（平均）：{train_errors}")
    logger.info(f"トレーニングデータの再構成誤差（最大値）：{train_errors_max}")

    if hasattr(model, 'reached_threshold') and model.reached_threshold:
        logger.info(f"閾値を下回るモデルを発見: units_1_3={unit_1_3}, units_2={unit_2}")
        return model, True
    else:
        logger.info("閾値を下回りませんでした"+ str(init_num+1) +"回目")  
        return model, False




#　初期値の割り振りを行う関数
def try_init_point(params, train_data, unit_1_3, unit_2, period_log):
    #初期値の割り振り

    #パラメータの取得
    num_retry = params["num_retry"]

    print(f"学習期間: {period_log}")
    
    # マルチプロセスでの実行
    with multiprocessing.Pool(processes=num_retry) as pool:
        result_iter = pool.imap_unordered(
        partial(run_one_seed, params, train_data, unit_1_3, unit_2, period_log),
        list(range(num_retry)) 
    )
        success_found = False

        for model, success in result_iter:
            if success:
                print("✅ 有効なモデルが見つかったので並列処理を中断します")
                success_found = True
                break
            else:
                print("閾値を下回りませんでした")
        # 全部失敗
        pool.close()
        pool.join()

        if success_found:
            return model, True

        else:
            print(f"全ての初期値で閾値を下回るモデルは見つかりませんでした (units_1_3={unit_1_3}, units_2={unit_2})")
            print("----------------------------------------------------------------------------------------------------------------")
            return model, False



#次のノード数を再帰的に探索する関数
def search_unit13_recursive(params, train_data, unit2, period_log ,lower_bound, upper_bound, best_model=None, best_unit13=None, min_total_units=float('inf')):
    """
    unit13を再帰的に探索する関数。
    lower_bound: 閾値を上回ったunit13（非適合）
    upper_bound: 閾値を下回ったunit13（適合）
    """
    # 打ち切り条件
    if upper_bound - lower_bound <= 1:
        return best_model, best_unit13, min_total_units

    mid_unit13 = (lower_bound + upper_bound) // 2


    print("----------------------------------------------------------------------------------------------------------------")
    print(f"[再帰探索] unit13={mid_unit13}, unit2={unit2}")
    model, flag_low_threshold = try_init_point(params, train_data, mid_unit13, unit2, period_log)
    total_units = mid_unit13 * 2 + unit2

    if flag_low_threshold:
        print(f"✅ 閾値を下回りました: unit13={mid_unit13}")
        if total_units < min_total_units:
            best_model = model
            best_unit13 = mid_unit13
            min_total_units = total_units

        # 再帰的に下限を更新    
        return search_unit13_recursive(params, train_data, unit2, period_log,lower_bound, mid_unit13, best_model, best_unit13, min_total_units)
    else:
        # 再帰的に上限を更新
        return search_unit13_recursive(params, train_data, unit2, period_log,mid_unit13, upper_bound, best_model, best_unit13, min_total_units)


            
# モデルのノード数を探索する関数
def search_optimal_units(params, train_data, period_log):
    # 初期値の設定
    best_model = None
    best_unit_1_3 = None
    best_unit_2 = None
    min_total_units = float("inf")

    #ハイパーパラメータの取得
    units = params["unit"]
    first_unit13 = units * 2
    max_unit2 = units - 1

    # ユニット数を保持する変数
    previous_best_unit13 = None  # unit2=10のときの最良unit13
    previous_unit2 = None

    #終わりのフラグ
    end_flag = False



    #探索開始
    for unit_2 in reversed(range(1,max_unit2+1)):
        print(f"\n===== unit2={unit_2} の探索開始 =====")
        #unit_2に対応する最適ユニット数、べストモデル、再構成誤差を記録する変数を作成

        if previous_best_unit13 is None:
            unit13 = first_unit13
        else:
            # 少数切り捨て
            unit13 = (previous_best_unit13 * 2 + previous_unit2 - unit_2) // 2
        print(f"unit2={unit_2} に対するunit13探索開始点: {unit13}")

        # 探索したユニット数を記録するセット
        searched_unit13 = set()

        #ノード探索に使用する変数
        lower = units+1
        upper = None


        # 各unit13に対するupperを定義する
        if unit_2 == max_unit2:
            unit13 = first_unit13
            while True:
                print("----------------------------------------------------------------------")
                print(f"[unit10探索] unit13={unit13}, unit2={unit_2}")

                searched_unit13.add(unit13)

                # モデルの学習
                model, flag_low_threshold = try_init_point(params, train_data, unit13, unit_2, period_log)

                if flag_low_threshold:
                    #upperの代入
                    upper = unit13
                    total_units = unit13 * 2 + unit_2
                    min_total_units = total_units
                    best_model = model
                    best_unit_1_3 = unit13
                    best_unit_2 = unit_2
                    print(f"🎉 最良モデル更新: unit13={unit13}, unit2={unit_2}, 合計={total_units}")
                    unit13 = (upper + lower) // 2  # 初期値を設定
                    break
                
                lower = unit13
                unit13 *= 2  # 見つかるまで倍々探索
                

        else:
            # unit2 < 10 のとき：upper は過去のベスト構成から計算
            upper = ((best_unit_1_3 * 2 + best_unit_2) - unit_2 ) // 2
            unit13 = upper # 初期値を設定


        # unit13の探索
        while True:
            # 収束するか、入力ノード数を下回ったら終了
            if unit13 in searched_unit13 or unit13 <= units:
                
                print(f"探索が収束 or 範囲外（unit13={unit13}）。unit2={unit_2} の探索終了。")
                break

            searched_unit13.add(unit13)
            print(f"[探索の開始] unit13={unit13}, unit2={unit_2}")

            #モデルの学習
            model, flag_low_threshold = try_init_point(params, train_data, unit13, unit_2, period_log)

            if flag_low_threshold == False and len(searched_unit13) == 1 and unit_2 != max_unit2:
                # 初回の探索で閾値を下回らなかった場合、探索を終了
                print(f"unit2={unit_2} に対するunit13={unit13} でも閾値を下回らなかったので以降の探索を完全に終了します")
                
                if best_model is None:
                    print("❗ 最適モデルが見つかりませんでした。最後に探索したモデルを返します。")
                    return model
                
                end_flag = True
                break

            #閾値を下回った場合、最適なユニット数を更新
            if flag_low_threshold == True:
                total_units = unit13 * 2 + unit_2
                if total_units < min_total_units:
                    min_total_units = total_units
                    best_model = model
                    best_unit_1_3 = unit13
                    best_unit_2 = unit_2
                    print(f"🎉 最良モデル更新: unit13={unit13}, unit2={unit_2}, 合計={total_units}")
                    print("------------------------------------------------------------------------------------------")
                upper = unit13

                # 探索の再帰的な呼び出し
                model, best_unit1_3, min_total_units = search_unit13_recursive(
                    params, train_data, unit_2, period_log,lower, upper, best_model, unit13, min_total_units
                )
                best_model = model
                best_unit_1_3 = best_unit1_3
                best_unit_2 = unit_2
                print("===========================================================================")
                print(f"✅ unit2={unit_2} に対して決定された unit13={best_unit1_3}")
                break
                
            else:
                lower = unit13
            

                model, best_unit1_3, min_total_units = search_unit13_recursive(
                    params, train_data, unit_2, period_log ,lower, upper, best_model, upper, min_total_units
                )
                best_model = model
                best_unit_1_3 = best_unit1_3
                best_unit_2 = unit_2
                print("===========================================================================")
                print(f"✅ unit2={unit_2} に対して決定された unit13={best_unit1_3}")
                break
                

        # unit2での探索が終了した場合、最適なユニット数を記録
        previous_best_unit13 = best_unit_1_3
        previous_unit2 = best_unit_2

        if end_flag:
            # 探索を終了するフラグが立った場合、ループを抜ける
            print("探索を終了します")
            break


    if best_model is None:
        print("❗ 最適モデルが見つかりませんでした。最後に探索したモデルを返します。")
        return model
    
    # 最適なユニット数を記録
    print("\n===== 最適構成の出力 =====")
    print(f"最適ノード数: unit13={best_unit_1_3}, unit2={best_unit_2}")
    print(f"合計ノード数: {min_total_units}")
    return best_model

        