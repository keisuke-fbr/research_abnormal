#実験が完了したときに各特徴量や閾値の学習がどのようになっているかを確認する

#必要ライブラリのインストール
import pandas as pd
import os

import tensorflow as tf

from functions import model_function
import config
import time





def result(data_ex , params):

    #処理時間の計測
    start = time.time()

    
    num = params["num"]
    

    #１，出力用のデータフレームを用意する
    thresholds_df = pd.DataFrame() 

    # 各期間で再学習するかどうかのフラグ
    flag_skip = 0

    #テストデータの異常スコア
    abnormal_scores = pd.DataFrame(columns=["measurement_date"] + ["anomaly_score"])
    #閾値
    periods = ["period1", "period2", "period3"]
    #特徴量ごとのトレーニングデータに対する再構成誤差の平均
    reconstraction_train_per_features = pd.DataFrame(index=periods, columns=config.columns_list)
    #特徴量ごとのテストデータに対する再構成誤差の平均
    reconstraction_test_per_features = pd.DataFrame(index=periods, columns=config.columns_list)
    #閾値データの特徴量ごとの再構成誤差
    thresholds_per_features = pd.DataFrame(index=periods, columns=config.columns_list)

    # トレーニング開始日の設定
    data_trainstart = params["train_start"]

    data_teststart = params["test_start"]

    # 開始日の年と月を抽出
    data_trainstart_dt = pd.to_datetime(data_trainstart)
    data_trainstart_year = data_trainstart_dt.year
    data_trainstart_month = data_trainstart_dt.month

    data_trainend_year = data_trainstart_year + 2
    data_trainend_month = data_trainstart_month
    data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

    # 開始日の年と月を抽出
    data_teststart_dt = pd.to_datetime(data_teststart)
    data_teststart_year = data_teststart_dt.year
    data_teststart_month = data_teststart_dt.month

    data_testend_year = data_teststart_year
    data_testend_month = data_teststart_month + 1
    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"
    
  

    #モデル実装部分
    for period in range(num):
        print("=====================================================================================")
        print(str(period+1) + "回目の期間のモデル")

        #i回目のデータの抜き出し
        #トレーニングデータの抜き出し
        train_data = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        # 日付の抜き出し
        measurement_date_train = train_data["measurement_date"]
        train_data = train_data[config.columns_list]
        # わずかな値を加え、最小値を０からずらす
        train_data += 0.0001
        #numpy.ndarrayに変換
        train_data = train_data.values
    
        #テストデータの抜き出し
        #テストデータはdata_trainendの一か月である
        test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        measurement_date_test = test_data["measurement_date"]
        test_data = test_data[config.columns_list]
        # わずかな値を加え、最小値を０からずらす
        test_data += 0.0001
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

        #モデルの作成およびモデル学習

        # 再学習の必要がある場合のみ学習を行う
        if flag_skip == 0:
            #各実験ごとの学習済みモデルと各特徴量ごとの学習過程配列が返される
            model = model_function.single_model(params,train_data,period)

            print(str(period+1) + "回目の期間のモデル作成終了")

        else:
            print(str(period+1) + "回目の期間においてはすでにあるモデルを使用します")

                         
        #データの計算
        train_predict_data = model.predict(train_data)
        test_predict_data = model.predict(test_data)


        #モデルを更新するなら他の必要データセットも更新するがそうでなければ余計な部分は更新しない
        if flag_skip == 0:
            #必要データセットの受け取り
            abnormal_score, threshold_data, errors_train_per_features_ave, errors_predict_per_features_ave, threshold_per_features, reconstract_error, errors_train_per_data = model_function.calcurate_data(train_data, test_data, train_predict_data, test_predict_data)
        else:
            abnormal_score, errors_predict_per_features_ave = model_function.calcurate_data(train_data, test_data, train_predict_data, test_predict_data, flag_skip)


        # データセットへの格納
        if flag_skip == 0:
            # データセットへの格納
            # 特徴量ごとのトレーニングデータに対する再構成誤差の平均
            reconstraction_train_per_features.loc["period"+str(period+1)] = errors_train_per_features_ave

            # 閾値データの特徴量ごとの再構成誤差
            thresholds_per_features.loc["period"+str(period+1)] = threshold_per_features

            #各データごとの再構成誤差
            errors_train_per_data_df = pd.DataFrame({
               "measurement_date": measurement_date_train,
                "errors_score": errors_train_per_data.numpy().flatten()
            })

            # 各期間の辞書
            thresholds_oneperiod = {
                "test_start": data_teststart,
                "test_end": data_testend,
                "threshold_data": threshold_data
            }  
            # 辞書を1行のDataFrameに変換してconcat
            thresholds_df = pd.concat([thresholds_df, pd.DataFrame([thresholds_oneperiod])], ignore_index=True)
       
        # 特徴量ごとのテストデータに対する再構成誤差の平均
        reconstraction_test_per_features.loc["period"+str(period+1)] = errors_predict_per_features_ave 
        
        #データセットをnumpy型からpandas型へ
        #異常スコア
        abnormal_score_df = pd.DataFrame({
            "measurement_date": measurement_date_test,
            "anomaly_score": abnormal_score
        })

        #　テストデータの異常度
        abnormal_scores = pd.concat([abnormal_scores,abnormal_score_df])

        #この時、abnormal_scoreのうちある値が閾値を超えているならば、次の期間においてはモデルの学習を行わず同じモデルを異常度の計算に使用する。
        if (abnormal_score > threshold_data).any() and flag_skip == 0:
            flag_skip = 1
            print("学習データの中に外れ値が発見されたので以降ではモデルの学習を行いません")
        else:
            print("次の期間でも学習を続けます")

        # ログの出力
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"{period+1}期間での結果の概要")
        print("トレーニングデータに対する再構成誤差の最大値かつ閾値")
        print(f"{threshold_data}")
        print(f"トレーニングデータに対する再構成誤差:{reconstract_error}")
        print(f"トレーニングデータに対する特徴量ごとの再構成誤差の平均")
        print(errors_train_per_features_ave)
        print(f"テストデータに対する特徴量ごとの再構成誤差の平均")
        print(errors_predict_per_features_ave)
        print("閾値データの各特徴量ごとの再構成誤差")
        print(threshold_per_features)

        

 
        print("trainの日付の最大値:" + str(data_trainend)) 
        print("testの日付の最大値:" + str(data_testend))

        #結果の保存

        print("異常値と閾値の保存")
        

        # 異常値の出力
        # 保存する相対パス
        relative_path = "result"
        # データをデータフレームに変換
        abnormal_score_oneperiod = pd.DataFrame(abnormal_score_df)
        os.makedirs(relative_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        # ファイル名を動的に生成（例: anomaly_data_2016-06-01.csv）
        data_teststart_dt = pd.to_datetime(data_teststart)
        file_name = f"anomaly_scores_{data_teststart_dt.strftime('%Y-%m-%d')}.csv"
        # フルパスを作成
        output_file = os.path.join(relative_path, file_name)
        # CSVファイルに保存
        abnormal_score_oneperiod.to_csv(output_file, index=False)


        #　閾値の出力
        # 保存する相対パス
        relative_path = "thresholds"
        # データをデータフレームに変換
        threshold_oneperiod_df = pd.DataFrame([thresholds_oneperiod])
        os.makedirs(relative_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        # ファイル名を動的に生成
        file_name = f"thresholds_scores_{data_teststart_dt.strftime('%Y-%m-%d')}.csv"
        # フルパスを作成
        output_file = os.path.join(relative_path, file_name)
        # CSVファイルに保存
        threshold_oneperiod_df.to_csv(output_file, index=False)


        #　特徴ごとの再構成誤差の出力
        # 保存する相対パス
        relative_path = "reconstraction_train_per_features"
        # データをデータフレームに変換
        errors_train_per_features_ave_df = pd.DataFrame([errors_train_per_features_ave])
        os.makedirs(relative_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        # ファイル名を動的に生成
        file_name = f"reconstraction_train_per_features_{data_teststart_dt.strftime('%Y-%m-%d')}.csv"
        # フルパスを作成
        output_file = os.path.join(relative_path, file_name)
        # CSVファイルに保存
        errors_train_per_features_ave_df.to_csv(output_file, index=False)   


        #各データごとの再構成誤差の出力
        # 保存する相対パス
        relative_path = "reconstraction_train_per_data"
        os.makedirs(relative_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        # ファイル名を動的に生成
        file_name = f"reconstraction_train_per_data_{data_teststart_dt.strftime('%Y-%m-%d')}.csv"
        # フルパスを作成
        output_file = os.path.join(relative_path, file_name)
        # CSVファイルに保存
        errors_train_per_data_df.to_csv(output_file, index=False, header=True) 



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

    abnormal_scores = pd.DataFrame(abnormal_scores)

    #時間の計測
    end = time.time()

    print(f"実行時間: {end - start:.6f} 秒")

    return  abnormal_scores, thresholds_df