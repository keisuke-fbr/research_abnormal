#実験が完了したときに各特徴量や閾値の学習がどのようになっているかを確認する

#必要ライブラリのインストール
import pandas as pd
import os

import tensorflow as tf

from src import model_basic
import config

def search_model(params,model, train_data):
    #１，変数の準備
    best_model = None
    best_loss = float('inf')

    #初期化回数
    num_retry = params["num_retry"]

    # シード値の抜き出し
    seeds = params["seeds"]
    learning_rate = params["learning_rate"]


    #初期値振りを4回行い学習を開始する
    for i in range(num_retry):
        
        print("============================================================================================")

        print(f"初期化：{i+1}回目")

        #２，シード値のセット
        model_basic.set_seed(seeds[i])
        print(f"シード値：{seeds[i]}")

        # オプティマイザを作成
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # オプティマイザの変数を初期化
        grads = [tf.zeros_like(var) for var in model.trainable_weights]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #３，学習を開始する
        model, leran_epoch_per_features = model_basic.learn_model(params,model,train_data,optimizer)

        #４，学習したmodelを用いてトレーニングデータに対する再構成誤差を計算する
        train_predict = model.predict(train_data, verbose = 0)

        #再構成値を計算する
        train_errors = model_basic.root_mean_squared_error(train_data, train_predict)
        #５，四回のうち最も良かったベスト特徴量ごとの学習記録とモデルを記録する
        if train_errors < best_loss:
            best_loss = train_errors
            best_model = model
            best_learn_epoch_per_features = leran_epoch_per_features
        
        return best_model, best_learn_epoch_per_features
    



def result(data_ex , params):

    num = params["num"]

    #１，出力用の配列を用意する

    #テストデータの異常スコア
    abnormal_scores = pd.DataFrame(columns=["measurement_date"] + ["anomaly_score"])
    #閾値
    periods = ["period1", "period2", "period3"]
    thresholds = {period : None for period in periods}
    #特徴量ごとの学習エポック
    learning_epochs_per_features_array = pd.DataFrame(index=periods, columns=config.columns_list)
    #特徴量ごとのトレーニングデータに対する再構成誤差の平均
    reconstraction_train_per_features = pd.DataFrame(index=periods, columns=config.columns_list)
    #特徴量ごとのテストデータに対する再構成誤差の平均
    reconstraction_test_per_features = pd.DataFrame(index=periods, columns=config.columns_list)
    #閾値データの特徴量ごとの再構成誤差
    thresholds_per_features = pd.DataFrame(index=periods, columns=config.columns_list)

   
    #２，トレーニングデータ、テストデータの呼び出し
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
    for period in range(num):
        print("=====================================================================================")
        print(str(period+1) + "回目の期間のモデル")

        #i回目のデータの抜き出し
        #トレーニングデータの抜き出し
        train_data = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        train_data = train_data[config.columns_list]
        #numpy.ndarrayに変換
        train_data = train_data.values
    
        #テストデータの抜き出し
        #テストデータはdata_trainendの一か月である
        test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        measurement_date_test = test_data["measurement_date"]
        test_data = test_data[config.columns_list]
        #numpy.ndarrayに変換
        test_data = test_data.values


        #該当データ期間
        print("該当期間の確認")
        print("はじまり(train)" + str(data_trainstart))
        print("終わり(train)" + str(data_trainend))
        print("はじまり(test)" + str(data_teststart))
        print("終わり(test)" + str(data_testend))
    

        #初期点を変更しながら、最適モデルを構築する
        print(str(period+1) + "回目の期間のモデル作成開始")
        print(str(period+1) + "回目の期間のトレーニングデータ数:" + str(len(train_data)))

        #モデルの作成
        #ハイパーパラメータの取得
        initializer = params["initializer"]
        units_1_3 = params["unit_paras"][period]["unit_1_3"]
        units_2 = params["unit_paras"][period]["unit_2"]
        print("モデルの作成")
        print(f"初期化方法：{initializer}、units_1_3:{units_1_3}、units_2:{units_2}")
        model = model_basic.model_autoencoder(initializer, units_1_3, units_2)

        #各実験ごとの学習済みモデルと各特徴量ごとの学習過程配列が返される
        model , learning_epochs_per_features = search_model(params,model,train_data)

        #最も良かったモデルを保存する
        base_dir= "..\..\cache\model"
        model_name= "unit_1_3_" + str(units_1_3) + "unit_2_" + str(units_2) + "perod_" + str(period+1)+ ".keras"
        save_path = os.path.join(base_dir,model_name)
        model.save(save_path)
        
        print(str(period+1) + "回目の期間のモデル作成終了")
        
        #データの計算
        train_predict_data = model.predict(train_data)
        test_predict_data = model.predict(test_data)
        #必要データセットの受け取り
        abnormal_score, threshold_data, errors_train_per_features_ave, errors_predict_per_features_ave, threshold_per_features = model_basic.calcurate_data(train_data, test_data, train_predict_data, test_predict_data)


        #データセットをnumpy型からpandas型へ
        #異常スコア
        abnormal_score_df = pd.DataFrame({
            "measurement_date": measurement_date_test,
            "anomaly_score": abnormal_score
        })

        # ログの出力
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"{period+1}期間での結果の概要")
        print(f"閾値：{threshold_data}")
        print(f"トレーニングデータに対する特徴量ごとの再構成誤差の平均")
        print(errors_train_per_features_ave)
        print(f"テストデータに対する特徴量ごとの再構成誤差の平均")
        print(errors_predict_per_features_ave)
        print("閾値データの各特徴量ごとの再構成誤差")
        print(threshold_per_features)
        print("学習終了時点の各特徴量の条件を満たしたエポック数")
        print(learning_epochs_per_features)

        # データセットへの格納
        #　テストデータの異常度
        abnormal_scores = pd.concat([abnormal_scores,abnormal_score_df])
        # 閾値
        thresholds["period"+str(period+1)] = {"test_start":data_teststart, "test_end":data_testend, "threshold_data": threshold_data}

        #　特徴量ごとの学習エポック
        learning_epochs_per_features_array.loc["period"+str(period+1)] = learning_epochs_per_features

        # 特徴量ごとのトレーニングデータに対する再構成誤差の平均
        reconstraction_train_per_features.loc["period"+str(period+1)] = errors_train_per_features_ave
       
        # 特徴量ごとのテストデータに対する再構成誤差の平均
        reconstraction_test_per_features.loc["period"+str(period+1)] = errors_predict_per_features_ave 

        # 閾値データの特徴量ごとの再構成誤差
        thresholds_per_features.loc["period"+str(period+1)] = threshold_per_features
        

 
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
    print("data_exのデータ数:" + str(len(data_ex)))
    # 日付が文字列で保存されている場合、日付型に変換します
    data_ex['measurement_date'] = pd.to_datetime(data_ex['measurement_date'])
    # 2019年2月のデータをフィルタリングして、そのデータ数をカウントします
    data_feb_2019 = data_ex[(data_ex['measurement_date'].dt.year == 2019) & (data_ex['measurement_date'].dt.month == 2)]
    # データ数を表示
    feb_2019_count = data_feb_2019.shape[0]
    print(f"data_exの最終月に含まれるデータ数:{feb_2019_count}")

    return  abnormal_scores, thresholds, learning_epochs_per_features_array, reconstraction_train_per_features, reconstraction_test_per_features, thresholds_per_features
        
        
    
    



  