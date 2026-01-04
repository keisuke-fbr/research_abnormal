# model_basic.py
# 実験が完了したときに各特徴量や閾値の学習がどのようになっているかを確認する

# 必要ライブラリのインストール
import pandas as pd
import numpy as np
import os
import time

import tensorflow as tf

from functions import model_function
import config


# =============================================================================
# ディレクトリ関連
# =============================================================================

def setup_output_directories(base_path, seeds):
    """
    出力用ディレクトリ構造の作成
    
    入力:
        base_path: str - ベースパス（data/）
        seeds: list - シード値リスト
    出力:
        なし
    """

    # 既存のディレクトリがある場合は、新規に作成されない
    # anomaly_score ディレクトリ
    os.makedirs(os.path.join(base_path, "anomaly_score", "anomaly_scores_per_period"), exist_ok=True)
    
    # threshold ディレクトリ
    os.makedirs(os.path.join(base_path, "threshold", "threshold_scores_per_period"), exist_ok=True)
    
    # learning_scores ディレクトリ（各シードごと）
    for seed in seeds:
        os.makedirs(os.path.join(base_path, "learning_scores", f"learning_scores_seed{seed}"), exist_ok=True)
    
    # errors ディレクトリ
    os.makedirs(os.path.join(base_path, "errors", "errors_average_per_features"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "errors", "errors_max_per_features"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "errors", "errors_per_data"), exist_ok=True)


# =============================================================================
# データ抽出関連
# =============================================================================

def extract_period_data(data_ex, data_original, train_start, train_end, test_start, test_end):
    """
    指定期間のトレーニング/テストデータを抽出
    
    入力:
        data_ex: DataFrame - 正規化済みデータ
        data_original: DataFrame - 正規化前データ
        train_start: str - トレーニング開始日
        train_end: str - トレーニング終了日
        test_start: str - テスト開始日
        test_end: str - テスト終了日
    出力:
        train_data: ndarray - トレーニングデータ（正規化済み）
        test_data: ndarray - テストデータ（正規化済み）
        train_data_original: ndarray - トレーニングデータ（正規化前）
        measurement_date_train: Series - トレーニングデータの日付
        measurement_date_test: Series - テストデータの日付
    """
    # トレーニングデータの抽出（正規化済み）
    train_df = data_ex[(data_ex["measurement_date"] < train_end) & 
                       (data_ex["measurement_date"] >= train_start)]
    measurement_date_train = train_df["measurement_date"]
    train_data = train_df[config.columns_list].copy()
    train_data = train_data.values
    
    # テストデータの抽出（正規化済み）
    test_df = data_ex[(data_ex["measurement_date"] < test_end) & 
                      (data_ex["measurement_date"] >= test_start)]
    measurement_date_test = test_df["measurement_date"]
    test_data = test_df[config.columns_list].copy()
    test_data = test_data.values
    
    # トレーニングデータの抽出（正規化前）- 基準値計算用
    train_original_df = data_original[(data_original["measurement_date"] < train_end) & 
                                       (data_original["measurement_date"] >= train_start)]
    train_data_original = train_original_df[config.columns_list].values
    
    return train_data, test_data, train_data_original, measurement_date_train, measurement_date_test


# =============================================================================
# 結果保存関連
# =============================================================================

def save_period_results(base_path, period_log, test_start_dt, abnormal_score_df, 
                        threshold_oneperiod, errors_train_per_features_avg, 
                        threshold_per_features, errors_train_per_data_df):
    """
    期間ごとの結果をCSVファイルに保存
    
    入力:
        base_path: str - ベースパス
        period_log: int - 期間番号
        test_start_dt: datetime - テスト開始日
        abnormal_score_df: DataFrame - 異常スコア
        threshold_oneperiod: dict - 閾値データ
        errors_train_per_features_avg: Series - 特徴量ごとの平均再構成誤差
        threshold_per_features: Series - 閾値データの特徴量ごとの再構成誤差
        errors_train_per_data_df: DataFrame - データごとの再構成誤差
    出力:
        なし
    """
    date_str = test_start_dt.strftime('%Y-%m-%d')
    
    # 異常スコアの保存
    anomaly_path = os.path.join(
        base_path, "anomaly_score", "anomaly_scores_per_period",
        f"anomaly_scores_{date_str}.csv"
    )
    abnormal_score_df.to_csv(anomaly_path, index=False)
    
    # 閾値の保存
    threshold_path = os.path.join(
        base_path, "threshold", "threshold_scores_per_period",
        f"threshold_scores_{date_str}.csv"
    )
    threshold_df = pd.DataFrame([threshold_oneperiod])
    threshold_df.to_csv(threshold_path, index=False)
    
    # 特徴量ごとの平均再構成誤差の保存
    errors_avg_path = os.path.join(
        base_path, "errors", "errors_average_per_features",
        f"errors_average_per_features_period{period_log}.csv"
    )
    errors_train_per_features_avg_df = pd.DataFrame([errors_train_per_features_avg])
    errors_train_per_features_avg_df.to_csv(errors_avg_path, index=False)
    
    # 閾値データの特徴量ごとの再構成誤差の保存
    errors_max_path = os.path.join(
        base_path, "errors", "errors_max_per_features",
        f"errors_max_per_features_period{period_log}.csv"
    )
    threshold_per_features_df = pd.DataFrame([threshold_per_features])
    threshold_per_features_df.to_csv(errors_max_path, index=False)


def save_combined_results(base_path, abnormal_scores, thresholds_df, errors_per_data_all):
    """
    全期間の結合ファイルを保存
    
    入力:
        base_path: str - ベースパス
        abnormal_scores: DataFrame - 全期間の異常スコア
        thresholds_df: DataFrame - 全期間の閾値
        errors_per_data_all: DataFrame - 全期間のデータごと再構成誤差
    出力:
        なし
    """
    # 結合異常スコアの保存
    combined_anomaly_path = os.path.join(
        base_path, "anomaly_score", "combined_anomaly_scores.csv"
    )
    abnormal_scores.to_csv(combined_anomaly_path, index=False)
    
    # 結合閾値の保存
    combined_threshold_path = os.path.join(
        base_path, "threshold", "combined_threshold_scores.csv"
    )
    thresholds_df.to_csv(combined_threshold_path, index=False)
    
    # 結合データごと再構成誤差の保存
    errors_per_data_path = os.path.join(
        base_path, "errors", "errors_per_data", "errors_per_data.csv"
    )
    errors_per_data_all.to_csv(errors_per_data_path, index=False)


# =============================================================================
# メイン実験ループ
# =============================================================================

def result(data_ex, data_original, params):
    """
    メイン実験ループ。期間ごとにモデル探索・学習・評価を実行
    
    入力:
        data_ex: DataFrame - 正規化済みデータ
        data_original: DataFrame - 正規化前データ（基準値計算用）
        params: dict - ハイパーパラメータ
    出力:
        abnormal_scores: DataFrame - 全期間の異常スコア
        thresholds_df: DataFrame - 全期間の閾値
    """
    # 処理時間の計測
    start = time.time()
    
    # パラメータの取得
    num = params["num"]
    
    # シード値リストの生成
    seeds = model_function.generate_seeds(params["seed_start"], params["num_retry"])
    
    # ベースパスの設定
    base_path = "data"
    learning_scores_base_path = os.path.join(base_path, "learning_scores")
    
    # 出力ディレクトリの作成
    setup_output_directories(base_path, seeds)
    
    # 出力用のデータフレームを用意する
    thresholds_df = pd.DataFrame()
    abnormal_scores = pd.DataFrame(columns=["measurement_date", "anomaly_score"])
    errors_per_data_all = pd.DataFrame()
    
    # 各期間で再学習するかどうかのフラグ
    flag_skip = 0
    
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
    
    # テスト開始日の年と月を抽出
    data_teststart_dt = pd.to_datetime(data_teststart)
    data_teststart_year = data_teststart_dt.year
    data_teststart_month = data_teststart_dt.month
    
    data_testend_year = data_teststart_year
    data_testend_month = data_teststart_month + 1
    if data_testend_month > 12:
        data_testend_year += 1
        data_testend_month = 1
    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"
    
    # モデル実装部分
    for period in range(num):
        period_log = period + 1
        
        # ログディレクトリの初期化
        model_function.initialize_log_directories(learning_scores_base_path, seeds, period_log)
        
        print("=====================================================================================")
        print(str(period_log) + "回目の期間のモデル")
        
        # 期間ごとのデータ抽出
        train_data, test_data, train_data_original, measurement_date_train, measurement_date_test = \
            extract_period_data(data_ex, data_original, data_trainstart, data_trainend, 
                               data_teststart, data_testend)
        
        # 該当データ期間
        print("該当期間の確認")
        print("はじまり(train)" + str(data_trainstart))
        print("終わり(train)" + str(data_trainend))
        print("はじまり(test)" + str(data_teststart))
        print("終わり(test)" + str(data_testend))
        
        # 初期点を変更しながら、最適モデルを構築する
        print(str(period_log) + "回目の期間のモデル作成開始")
        print(str(period_log) + "回目の期間のトレーニングデータ数:" + str(len(train_data)))
        
        # 再学習の必要がある場合のみ学習を行う
        if flag_skip == 0:
            # 基準値の計算
            if params.get("test_mode", False):
                # テストモード：固定基準値を使用
                thresholds = model_function.generate_test_thresholds(
                    config.columns_list, params["test_threshold"]
                )
            else:
                # 通常モード：データから基準値を計算
                thresholds = model_function.calculate_thresholds(train_data_original, config.columns_list)    
            # 各実験ごとの学習済みモデルが返される
            model = model_function.search_optimal_units(
                params, train_data, period_log, thresholds, learning_scores_base_path
            )
            
            print(str(period_log) + "回目の期間のモデル作成終了")
        else:
            print(str(period_log) + "回目の期間においてはすでにあるモデルを使用します")
        
        # データの計算
        train_predict_data = model.predict(train_data)
        test_predict_data = model.predict(test_data)
        
        # モデルを更新するなら他の必要データセットも更新する
        if flag_skip == 0:
            # 必要データセットの受け取り
            (abnormal_score, threshold_data, errors_train_per_features_avg, 
             errors_predict_per_features_avg, threshold_per_features, 
             reconstract_error, errors_train_per_data) = model_function.calculate_data(
                train_data, test_data, train_predict_data, test_predict_data
            )
        else:
            abnormal_score, errors_predict_per_features_avg = model_function.calculate_data(
                train_data, test_data, train_predict_data, test_predict_data, flag_skip
            )
        
        # データセットへの格納
        if flag_skip == 0:
            # 各データごとの再構成誤差
            errors_train_per_data_df = pd.DataFrame({
                "measurement_date": measurement_date_train,
                "errors_score": errors_train_per_data.numpy().flatten()
            })
            
            # 全期間のデータごと再構成誤差に追加
            errors_per_data_all = pd.concat([errors_per_data_all, errors_train_per_data_df], ignore_index=True)
            
            # 各期間の閾値辞書
            threshold_oneperiod = {
                "test_start": data_teststart,
                "test_end": data_testend,
                "threshold_data": threshold_data
            }
            
            # 辞書を1行のDataFrameに変換してconcat
            thresholds_df = pd.concat([thresholds_df, pd.DataFrame([threshold_oneperiod])], ignore_index=True)
            
            # 期間ごとの結果を保存
            test_start_dt = pd.to_datetime(data_teststart)
            abnormal_score_df = pd.DataFrame({
                "measurement_date": measurement_date_test,
                "anomaly_score": abnormal_score
            })
            
            save_period_results(
                base_path, period_log, test_start_dt, abnormal_score_df,
                threshold_oneperiod, errors_train_per_features_avg,
                threshold_per_features, errors_train_per_data_df
            )
        else:
            # flag_skip == 1 の場合も異常スコアは保存
            test_start_dt = pd.to_datetime(data_teststart)
            abnormal_score_df = pd.DataFrame({
                "measurement_date": measurement_date_test,
                "anomaly_score": abnormal_score
            })
            
            date_str = test_start_dt.strftime('%Y-%m-%d')
            anomaly_path = os.path.join(
                base_path, "anomaly_score", "anomaly_scores_per_period",
                f"anomaly_scores_{date_str}.csv"
            )
            abnormal_score_df.to_csv(anomaly_path, index=False)
        
        # テストデータの異常度を集約
        abnormal_scores = pd.concat([abnormal_scores, abnormal_score_df], ignore_index=True)
        
        # 異常判定：テストデータの中に閾値を超えるデータが確認されると以降は再学習しない
        if flag_skip == 0 and (abnormal_score > threshold_data).any():
            flag_skip = 1
            print("学習データの中に外れ値が発見されたので以降ではモデルの学習を行いません")
        else:
            print("次の期間でも学習を続けます")
        
        # ログの出力
        if flag_skip == 0 or period == 0:
            print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"{period_log}期間での結果の概要")
            print("トレーニングデータに対する再構成誤差の最大値かつ閾値")
            print(f"{threshold_data}")
            print(f"トレーニングデータに対する再構成誤差:{reconstract_error}")
            print(f"トレーニングデータに対する特徴量ごとの再構成誤差の平均")
            print(errors_train_per_features_avg)
            print(f"テストデータに対する特徴量ごとの再構成誤差の平均")
            print(errors_predict_per_features_avg)
            print("閾値データ（最大の再構成誤差）の各特徴量ごとの再構成誤差")
            print(threshold_per_features)
        
        print("trainの日付の最大値:" + str(data_trainend))
        print("testの日付の最大値:" + str(data_testend))
        print("異常値と閾値の保存")
        
        # 日付の更新（1ヶ月ずらす）
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
    
    # 結合ファイルの保存
    save_combined_results(base_path, abnormal_scores, thresholds_df, errors_per_data_all)
    
    # データの確認ログ
    print("data_exの日付の最大値:" + str(data_ex["measurement_date"].max()))
    print("data_exのデータ数:" + str(len(data_ex)))
    
    # 日付が文字列で保存されている場合、日付型に変換
    data_ex_copy = data_ex.copy()
    data_ex_copy['measurement_date'] = pd.to_datetime(data_ex_copy['measurement_date'])
    
    # 最終月のデータをフィルタリングして、そのデータ数をカウント
    max_date = data_ex_copy['measurement_date'].max()
    data_final_month = data_ex_copy[
        (data_ex_copy['measurement_date'].dt.year == max_date.year) & 
        (data_ex_copy['measurement_date'].dt.month == max_date.month)
    ]
    final_month_count = data_final_month.shape[0]
    print(f"data_exの最終月に含まれるデータ数:{final_month_count}")
    
    # 時間の計測
    end = time.time()
    print(f"実行時間: {end - start:.6f} 秒")
    
    return abnormal_scores, thresholds_df