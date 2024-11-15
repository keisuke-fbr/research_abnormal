#必要ライブラリのインストール
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




#モデルの最小化関数
# カスタム損失関数の定義
def custom_loss(delta):
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        small_error_loss = tf.square(error)
        large_error_loss = error
        return tf.where(error < delta, large_error_loss, small_error_loss)
    return loss


# シード値を決定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def add_to_dict_with_average(dictionary, key, value):
    if key in dictionary:
        # 既存の値がリストでない場合、リストに変換
        if not isinstance(dictionary[key], list):
            dictionary[key] = [dictionary[key]]  # 値をリストに変換
        # リストに値を追加
        dictionary[key].append(value)
        # 平均値を再計算して辞書に格納
        dictionary[key] = np.mean(dictionary[key])
    else:
        # キーが存在しない場合、新しいリストとして値を追加
        dictionary[key] = value


def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

#異常スコア算出関数
def abnomalScores(test_data,predicted_data):
  # 平均二乗誤差 (MSE) を各サンプルに対して計算
    mse_per_sample = np.mean(np.square(test_data - predicted_data), axis=1)

    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_sample = np.sqrt(mse_per_sample)

    return rmse_per_sample




#閾値を求める関数
def biggest_threshold(train_data, traindata_model):
    # 平均二乗誤差 (MSE) を各サンプルに対して計算
    mse_per_sample = np.mean(np.square(train_data - traindata_model), axis=1)

    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 最も大きなRMSEを取得
    max_rmse = np.max(rmse_per_sample)

    return max_rmse

#閾値となったデータの各特徴量ごとの誤差を算出するプログラム
def biggest_threshold_per_features(train_data, traindata_model):
    # 平均二乗誤差 (MSE) を各サンプルに対して計算
    mse_per_sample = np.mean(np.square(train_data - traindata_model), axis=1)
    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_sample = np.sqrt(mse_per_sample)
    # 最も大きなRMSEを取得
    max_rmse = np.max(rmse_per_sample)
    #その要素番号を取得
    max_rmse_index = np.argmax(rmse_per_sample)

    print(f"要素番号:{max_rmse_index}")

    print("train_data shape:", train_data.shape)
    print("traindata_model shape:", traindata_model.shape)


    #その要素番号の特徴量を取得する
    max_features = np.sqrt(np.square(train_data.iloc[max_rmse_index] - traindata_model.iloc[max_rmse_index]))


    return max_features


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
    model.add(layers.Dense(units=middle_unit_1, activation="sigmoid" ,input_shape=(input_unit,), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5))))
    # 中間層２の作成
    model.add(layers.Dense(units=middle_unit_2, activation="sigmoid", kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev = pow(units_2, -0.5))))
    # 中間層３の作成
    model.add(layers.Dense(units=middle_unit_3, activation="sigmoid", kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5))))

    #出力層の作成
    model.add(layers.Dense(units=output_unit, activation="linear", kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev = pow(175, -0.5))))
    
    #最適化手法
    learning_rate = 1e-4
    #opt=tf.keras.optimizers.legacy.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False) # default

    

    model.compile(optimizer="adam",
                        loss=root_mean_squared_error
                       )  

    return model


#モデル探索の学習関数、モデルと学習データを入力することで学習を完了させ、その損失が得られる
#学習アルゴリズムはmax_epochs回学習させ、損失関数に変化がなくなった時点で学習を終了させる
def train_and_evaluate(model, X_train, max_epochs, early_stopping_params):

    early_stopping = EarlyStopping(**early_stopping_params)

    history = 0
    #ここでの終了条件はあくまでも収束条件
    history = model.fit(X_train, X_train, epochs=max_epochs, batch_size=64, verbose=0
                        , shuffle = True, validation_data = (X_train,X_train) ,callbacks=[early_stopping])

    # EarlyStoppingで打ち切られたかどうかを確認する
    if len(history.epoch) < max_epochs:
        print(f"モデルは収束しました。訓練は {len(history.epoch)} エポックで停止しました。")
    else:
        print("モデルは最大エポック数まで収束条件に達しませんでした。収束条件を満たさず終了しました。")

    #最終モデルの決定値は損失そのもの
    #損失関数はroot_mean_squared_errorであり、今回はバッチサイズが64なので64データの損失の平均を取得する
    final_loss = history.history['loss'][-1]  # 最終エポックの検証損失
    
    
    return final_loss


def reconstruction_error_per_feature(test_data, predicted_data):
    """
    オートエンコーダの学習後に、各特徴量ごとの再構成誤差を計算する関数。

    Parameters:
    - test_data: オリジナルのテストデータ（入力データ）
    - predicted_data: オートエンコーダによって再構成されたデータ

    Returns:
    - feature_errors: 各特徴量ごとの再構成誤差
    """
    # 各特徴量ごとの再構成誤差を計算
    feature_errors = np.square(test_data - predicted_data)

    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_features = np.sqrt(feature_errors)

    # 特徴量ごとの誤差を平均して出力
    mean_feature_errors = np.mean(rmse_per_features, axis=0)
    

    return mean_feature_errors


