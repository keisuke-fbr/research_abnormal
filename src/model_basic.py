#必要ライブラリのインストール
#必要ライブラリのインストール

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, losses, initializers

import config



# シード値を決定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#データの再構成値を計算する関数
def root_mean_squared_error(y_true, y_pred):
    #機械学習実装時の損失関数として用いるためバッチサイズに対して平均を取るような値を返す
    errors = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true),axis=1))
    errors = tf.reduce_mean(errors)
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

    #1行11列にする
    mean_errors_per_feature = mean_errors_per_feature.reshape(1,-1)
    
    return mean_errors_per_feature




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
    #opt=tf.keras.optimizers.legacy.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False) # default

    model.compile(optimizer="adam",
                        loss=root_mean_squared_error
                       )  

    return model

#モデルの重みの更新の高速化のため、tf.functionで定義
@tf.function
def train_step(x_batch, model, optimizer):
    with tf.GradientTape() as tape:
        #　７，その時点での予測値を算出する
        reconstructed = model(x_batch, training=True)
        #８，予測値から再構成誤差を算出する

        loss = root_mean_squared_error(x_batch, reconstructed)
    #９，再構成誤差から勾配を計算する
    grads = tape.gradient(loss, model.trainable_weights)
    #１０，勾配を計算し重みを更新する
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return 


# 実際に学習を行う関数
def learn_model(params, model , train_data, optimizer):
    #１，ハイパーパラメータをばらす
    batch_size = params["batch_size"]
    patience = params["patience"]
    min_delta = params["min_delta"]
    max_epochs = params["max_epochs"]
    detail_log = params["detail_log"]

    #２，計算のための配列を準備する
    #全体の再構成誤差を２回分保持しておく配列
    value_reconstractions = [float("inf"), float("inf")]
    # 全体の変化分を保持しておく変数
    value_change = 0
    when = ["now","before"]
    #各特徴量ごとの再構成誤差の値を２回分保持しておく配列
    value_reconstractions_per_features = pd.DataFrame(float("inf"),index=when, columns=config.columns_list)
    #各特徴量ごとの変化分を保持しておく配列
    value_change_per_features = pd.DataFrame(columns=config.columns_list)
    #閾値を何回満たしているのかを保持する変数
    count = 0
    #２，各特徴量ごとのエポックを記録する配列を準備する
    learning_epoch_per_features = pd.Series(0, index=config.columns_list)


    #いつ学習が完了したかのflag
    flag_early_stop = False

    #学習速度高速化のためtf形式に変更する
    # 2. データ型の統一
    train_data = tf.cast(train_data, dtype=tf.float32)  # TensorFlow用に型変換
    # データセットの作成
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # モデルの初期化（ダミー入力）
    dummy_input = tf.zeros((1, train_data.shape[1]), dtype=tf.float32)
    _ = model(dummy_input)


    #４，エポックごとにループを回す
    for epoch in range(max_epochs):
        #５，バッチ単位でループを回す
        for x_batch in dataset:
            #modelの更新
            train_step(x_batch, model, optimizer)
        
        #バッチサイズの出力
        len_train = len(train_data)
        len_batch = len_train//batch_size
        

        #１１，１エポック終了後に予測値を算出する
        train_predict_data = model(train_data,training = False)
        # １２，モデル全体の再構成誤差の計算
        errors = root_mean_squared_error(train_data, train_predict_data)
        # １３，モデル全体の再構成誤差を適切に配列に格納する
        value_reconstractions[1] = value_reconstractions[0]
        value_reconstractions[0] = errors

        # 全体の再構成誤差の変化分を計算する
        value_change = value_reconstractions[1] - value_reconstractions[0]


        # １３，各特徴量ごとの再構成誤差を計算する
        feature_errors = calcurate_reconstraction_error_per_features(train_data,train_predict_data)

        # その平均値をとる

        #１４，各特徴量ごとの再構成誤差を適切に配列に格納する
        
        value_reconstractions_per_features.loc["before"] = value_reconstractions_per_features.loc["now"]
        value_reconstractions_per_features.loc["now"] = feature_errors

        #１５，各特徴量ごとに再構成誤差の変化分を計算する
        value_change_per_features = value_reconstractions_per_features.loc["before"] - value_reconstractions_per_features.loc["now"]

        # ログの排出
        if detail_log:
            print(f"バッチ数：{len_batch}")
            print(f"{epoch+1}/{max_epochs}での再構成誤差：{errors},変化量：{value_change}")
            print("-----------------------------------------------------------------------------------")


        #１６，全体の再構成誤差の変化値が閾値を満たしているか確認する
        if value_change < min_delta:
            count += 1
        else:
            count = 0
        
        #１７，各特徴量ごとに再構成誤差の変化値が閾値を満たしているか確認する
        for feature in value_change_per_features.index:
            if abs(value_change_per_features[feature]) < min_delta:
                learning_epoch_per_features[feature] += 1
            else:
                learning_epoch_per_features[feature] = 0

        #countが1000になるのならば学習を終了する
        if count == patience:
            print(f"{patience}回条件を満たしたので学習を終了します")
            flag_early_stop = True

            break

        
    if flag_early_stop == False:
        print(f"{patience}回条件を満たさなかったので{max_epochs}回学習を行いました")
    
    return model, learning_epoch_per_features



def calcurate_data(train_data, test_data, train_predict_data, test_predict_data):
    #基本データの計算
    errors_train = root_mean_squared_error_per_data(train_data, train_predict_data)
    errors_predict = root_mean_squared_error_per_data(test_data, test_predict_data)

    errors_train_per_features = calcurate_reconstraction_error_per_features(train_data,train_predict_data)
    errors_predict_per_features = calcurate_reconstraction_error_per_features(test_data,test_predict_data)

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
    print(max_position)
    print(errors_per_data_per_features.shape)
    thresold_per_features = errors_per_data_per_features.iloc[max_position]

    return abnormal_score, thresold, errors_train_per_features_ave, errors_predict_per_features_ave, thresold_per_features
