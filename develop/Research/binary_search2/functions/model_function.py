# model_function.py
# オートエンコーダを使用した異常検知システムのコア機能

# 必要ライブラリのインポート
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from keras.saving import register_keras_serializable

import os
import multiprocessing
from functools import partial

from custom_Class.custom_earlystopping import CustomAutoencoder, MaxReconstructionErrorEarlyStopping
import config


# =============================================================================
# シード値関連
# =============================================================================

def set_seed(seed):
    """
    再現性のためのシード値設定
    
    入力:
        seed: int - シード値
    出力:
        なし
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_seeds(seed_start, num_retry):
    """
    シード値リストを自動生成
    
    入力:
        seed_start: int - シード値の最初の値
        num_retry: int - 初期値振り直し回数
    出力:
        seeds: list - シード値リスト
    """
    return [seed_start + i for i in range(num_retry)]


# =============================================================================
# 基準値計算関連
# =============================================================================

def calculate_significant_digits(value):
    """
    数値の有効桁数を計算（整数部 + 小数部）
    
    入力:
        value: float - 計算対象の値
    出力:
        significant_digits: int - 有効桁数
    """
    if value == 0:
        return 1
    
    abs_value = abs(value)
    
    # 文字列に変換して桁数を数える
    value_str = str(abs_value)
    
    # 小数点を除去して桁数をカウント
    if '.' in value_str:
        # 小数点を除いた数字の桁数
        significant_digits = len(value_str) - 1  # 小数点の1文字分を引く
    else:
        # 整数の場合
        significant_digits = len(value_str)
    
    return significant_digits


def calculate_thresholds(train_data_original, columns_list):
    """
    各特徴量の基準値を計算
    
    入力:
        train_data_original: ndarray - 正規化前トレーニングデータ
        columns_list: list - 特徴量名リスト
    出力:
        thresholds: dict - 特徴量名→基準値のマッピング
    """
    thresholds = {}
    
    for i, col_name in enumerate(columns_list):
        # 各特徴量の最大値を取得
        max_value = np.max(np.abs(train_data_original[:, i]))
        
        # 有効桁数を計算
        significant_digits = calculate_significant_digits(max_value)
        
        # 基準値 = 10^(-有効桁数)
        threshold = 10 ** (-significant_digits)
        thresholds[col_name] = threshold

        # デバッグ用出力
        print(f"特徴量: {col_name}, 最大値: {max_value}, 有効桁数: {significant_digits}, 基準値: {threshold}")
    
    return thresholds

def generate_test_thresholds(columns_list, test_threshold):
    """
    テストモード用の固定基準値を生成
    
    入力:
        columns_list: list - 特徴量名リスト
        test_threshold: float - 全特徴共通の基準値
    出力:
        thresholds: dict - 特徴量名→基準値のマッピング
    """
    thresholds = {}
    for col_name in columns_list:
        thresholds[col_name] = test_threshold
    print(f"[テストモード] 全特徴の基準値を {test_threshold} に設定")
    return thresholds
# =============================================================================
# 誤差計算関連
# =============================================================================

@register_keras_serializable()
def root_mean_squared_error(y_true, y_pred):
    """
    全体のRMSE計算（損失関数用）
    
    入力:
        y_true: Tensor - 正解データ
        y_pred: Tensor - 予測データ
    出力:
        rmse: Tensor - RMSE値（スカラー）
    """
    mse_total = tf.reduce_mean(tf.square(y_pred - y_true))
    rmse_total = tf.sqrt(mse_total)
    return rmse_total


def root_mean_squared_error_per_data(y_true, y_pred):
    """
    データごとのRMSE計算
    
    入力:
        y_true: Tensor - 正解データ
        y_pred: Tensor - 予測データ
    出力:
        rmse_per_data: Tensor - 各データのRMSE（形状: [N, 1]）
    """
    mse_per_sample = tf.reduce_mean(tf.square(y_pred - y_true), axis=1)
    rmse_per_sample = tf.sqrt(mse_per_sample)
    errors = tf.reshape(rmse_per_sample, (-1, 1))
    return errors


def root_mean_squared_error_per_feature(y_true, y_pred):
    """
    特徴量ごとのRMSE計算
    
    入力:
        y_true: ndarray - 正解データ
        y_pred: ndarray - 予測データ
    出力:
        rmse_per_feature: ndarray - 各特徴量のRMSE（形状: [特徴数]）
    """
    # 各特徴量ごとの誤差を計算
    errors = np.sqrt((y_true - y_pred) ** 2)
    # 各特徴量ごとに平均を取る
    mean_errors_per_feature = np.mean(errors, axis=0)
    return mean_errors_per_feature


def root_mean_squared_error_per_feature_per_data(y_true, y_pred):
    """
    各データの各特徴量ごとのRMSE計算
    
    入力:
        y_true: ndarray - 正解データ
        y_pred: ndarray - 予測データ
    出力:
        errors: ndarray - 各データの各特徴量のRMSE（形状: [N, 特徴数]）
    """
    errors = np.sqrt((y_true - y_pred) ** 2)
    return errors


# =============================================================================
# データ計算関連
# =============================================================================

def calculate_data(train_data, test_data, train_predict_data, test_predict_data, flag_skip=0):
    """
    各種誤差・異常スコアを一括計算
    
    入力:
        train_data: ndarray - トレーニングデータ
        test_data: ndarray - テストデータ
        train_predict_data: ndarray - トレーニング予測データ
        test_predict_data: ndarray - テスト予測データ
        flag_skip: int - スキップフラグ（デフォルト: 0）
    出力（flag_skip=0の場合）:
        abnormal_score: ndarray - テストデータの異常スコア
        threshold: float - 閾値
        errors_train_per_features_avg: Series - 特徴量ごとの平均再構成誤差
        errors_predict_per_features_avg: Series - テストの特徴量ごとの平均再構成誤差
        threshold_per_features: Series - 閾値データの特徴量ごと再構成誤差
        reconstruct_error: float - トレーニングデータ全体の再構成誤差
        errors_train_per_data: Tensor - トレーニングデータごとの再構成誤差
    出力（flag_skip=1の場合）:
        abnormal_score: ndarray - テストデータの異常スコア
        errors_predict_per_features_avg: Series - テストの特徴量ごとの平均再構成誤差
    """
    # 基本データの計算
    errors_train = root_mean_squared_error_per_data(train_data, train_predict_data)
    errors_predict = root_mean_squared_error_per_data(test_data, test_predict_data)
    
    # テストデータの異常スコア
    abnormal_score = errors_predict.numpy().flatten()
    
    # 特徴量ごとのテストデータに対する再構成誤差の平均
    errors_predict_per_features = root_mean_squared_error_per_feature(test_data, test_predict_data)
    errors_predict_per_features_avg = pd.Series(errors_predict_per_features, index=config.columns_list)
    
    if flag_skip == 1:
        return abnormal_score, errors_predict_per_features_avg
    
    # トレーニングデータに対する再構成誤差の計算
    reconstruct_error = root_mean_squared_error(train_data, train_predict_data).numpy()
    
    # 各データごとの再構成誤差
    errors_train_per_data = errors_train
    
    # 特徴量ごとのトレーニングデータに対する再構成誤差
    errors_train_per_features = root_mean_squared_error_per_feature(train_data, train_predict_data)
    errors_train_per_features_avg = pd.Series(errors_train_per_features, index=config.columns_list)
    
    # 各データ、各特徴量の再構成誤差
    errors_per_data_per_features = root_mean_squared_error_per_feature_per_data(train_data, train_predict_data)
    errors_per_data_per_features = pd.DataFrame(errors_per_data_per_features, columns=config.columns_list)
    
    # 閾値の計算（トレーニングデータの最大再構成誤差）
    threshold = errors_train.numpy().max()
    
    # 閾値データの特徴量ごとの再構成誤差
    max_position = int(tf.argmax(errors_train).numpy())
    print(f"最も再構成誤差が大きかったインデックス番号：{max_position}")
    threshold_per_features = errors_per_data_per_features.iloc[max_position]
    
    return (abnormal_score, threshold, errors_train_per_features_avg, 
            errors_predict_per_features_avg, threshold_per_features, 
            reconstruct_error, errors_train_per_data)


# =============================================================================
# モデル構築関連
# =============================================================================

def model_autoencoder(params, seed, unit13, unit2):
    """
    3層オートエンコーダモデル構築
    
    入力:
        params: dict - ハイパーパラメータ
        seed: int - シード値
        unit13: int - 中間層1, 3のユニット数
        unit2: int - 中間層2のユニット数
    出力:
        model: CustomAutoencoder - コンパイル済みモデル
    """
    # ハイパーパラメータの取得
    learning_rate = params["learning_rate"]
    input_unit = params["unit"]
    
    # シード値のセット
    set_seed(seed)
    
    # ユニット数の定義
    middle_unit_1 = unit13
    middle_unit_2 = unit2
    middle_unit_3 = unit13
    output_unit = input_unit
    
    # モデル構造の定義（Functional API）
    inputs = keras.Input(shape=(input_unit,))
    
    x = layers.Dense(
        units=middle_unit_1, 
        activation="sigmoid",
        kernel_initializer=initializers.TruncatedNormal(
            mean=0.0, stddev=pow(unit13, -0.5), seed=seed
        )
    )(inputs)
    
    x = layers.Dense(
        units=middle_unit_2, 
        activation="sigmoid",
        kernel_initializer=initializers.TruncatedNormal(
            mean=0.0, stddev=pow(unit2, -0.5), seed=seed
        )
    )(x)
    
    x = layers.Dense(
        units=middle_unit_3, 
        activation="sigmoid",
        kernel_initializer=initializers.TruncatedNormal(
            mean=0.0, stddev=pow(unit13, -0.5), seed=seed
        )
    )(x)
    
    outputs = layers.Dense(
        units=output_unit, 
        activation="linear",
        kernel_initializer=initializers.TruncatedNormal(
            mean=0.0, stddev=pow(175, -0.5), seed=seed
        )
    )(x)
    
    model = CustomAutoencoder(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error)
    
    return model


# =============================================================================
# ログ関連
# =============================================================================

def initialize_log_directories(base_path, seeds, period_log):
    """
    学習ログ用ディレクトリ初期化
    
    入力:
        base_path: str - ベースパス
        seeds: list - シード値リスト
        period_log: int - 期間番号
    出力:
        なし
    """
    for seed in seeds:
        dir_path = os.path.join(
            base_path, 
            f"learning_scores_seed{seed}",
            f"learning_scores_seed{seed}_period{period_log}"
        )
        os.makedirs(dir_path, exist_ok=True)


def save_learning_scores(log_data, base_path, seed, period_log, unit13, unit2):
    """
    学習ログをCSVに保存
    
    入力:
        log_data: dict - エポックごとの学習ログ
        base_path: str - ベースパス
        seed: int - シード値
        period_log: int - 期間番号
        unit13: int - 中間層1, 3のユニット数
        unit2: int - 中間層2のユニット数
    出力:
        なし
    """
    if not log_data or len(log_data["epoch"]) == 0:
        return
    
    # DataFrameに変換
    df = pd.DataFrame(log_data)
    
    # ファイルパス生成
    dir_path = os.path.join(
        base_path,
        f"learning_scores_seed{seed}",
        f"learning_scores_seed{seed}_period{period_log}"
    )
    file_name = f"learning_scores_seed{seed}_period{period_log}_unit13_{unit13}_unit2_{unit2}.csv"
    file_path = os.path.join(dir_path, file_name)
    
    # 保存
    df.to_csv(file_path, index=False)


# =============================================================================
# モデル学習関連
# =============================================================================

def learn_model(params, model, train_data, thresholds, seed, period_log, unit13, unit2, base_path):
    """
    モデル学習実行
    
    入力:
        params: dict - ハイパーパラメータ
        model: CustomAutoencoder - モデル
        train_data: ndarray - トレーニングデータ
        thresholds: dict - 特徴量ごとの基準値
        seed: int - シード値
        period_log: int - 期間番号
        unit13: int - 中間層1, 3のユニット数
        unit2: int - 中間層2のユニット数
        base_path: str - ベースパス
    出力:
        model: CustomAutoencoder - 学習済みモデル
    """
    # ハイパーパラメータをばらす
    batch_size = params["batch_size"]
    max_epochs = params["max_epochs"]
    log_counter = params["log_counter"]
    detail_log = params["detail_log"]
    
    # early_stoppingのパラメータ設定
    early_stopping_params = {
        "detail_log": detail_log,
        "thresholds": thresholds,
        "log_counter": log_counter
    }
    
    # early_stoppingの設定
    early_stopping = MaxReconstructionErrorEarlyStopping(
        model, early_stopping_params,
        seed=seed, period_log=period_log, unit13=unit13, unit2=unit2, base_path=base_path
    )
    
    # 訓練データをモデルに渡す
    model.train_data_for_monitoring = train_data
    
    # 学習速度高速化のためtf形式に変更
    train_data_tf = tf.cast(train_data, dtype=tf.float32)
    
    # 学習の開始
    history = model.fit(
        train_data_tf, train_data_tf, 
        epochs=max_epochs, 
        batch_size=batch_size, 
        verbose=0,
        shuffle=True, 
        validation_data=(train_data_tf, train_data_tf),
        callbacks=[early_stopping]
    )
    
    return model


# =============================================================================
# 並列処理関連
# =============================================================================

def run_one_seed(params, train_data, unit13, unit2, period_log, thresholds, base_path, init_num):
    """
    1つのシード値での学習実行（並列処理用）
    
    入力:
        params: dict - ハイパーパラメータ
        train_data: ndarray - トレーニングデータ
        unit13: int - 中間層1, 3のユニット数
        unit2: int - 中間層2のユニット数
        period_log: int - 期間番号
        thresholds: dict - 特徴量ごとの基準値
        base_path: str - ベースパス
        init_num: int - 初期化番号（0〜num_retry-1）
    出力:
        model: CustomAutoencoder - 学習済みモデル
        success: bool - 基準値を満たしたか
        seed: int - 使用したシード値
    """
    # seedの取り出し
    seeds = generate_seeds(params["seed_start"], params["num_retry"])
    seed = seeds[init_num]
    set_seed(seed)
    
    # モデル構築
    model = model_autoencoder(params, seed, unit13, unit2)
    
    # モデル学習
    model = learn_model(
        params, model, train_data, thresholds,
        seed, period_log, unit13, unit2, base_path
    )
    
    # 成功判定
    if hasattr(model, 'reached_threshold') and model.reached_threshold:
        return model, True, seed
    else:
        return model, False, seed


def try_init_point(params, train_data, unit13, unit2, period_log, thresholds, base_path):
    """
    複数シードでの並列学習・成功判定
    
    入力:
        params: dict - ハイパーパラメータ
        train_data: ndarray - トレーニングデータ
        unit13: int - 中間層1, 3のユニット数
        unit2: int - 中間層2のユニット数
        period_log: int - 期間番号
        thresholds: dict - 特徴量ごとの基準値
        base_path: str - ベースパス
    出力:
        model: CustomAutoencoder - 成功したモデル（または最後のモデル）
        success: bool - 基準値を満たしたか
    """
    num_retry = params["num_retry"]
    
    print(f"学習期間: {period_log}")
    
    # マルチプロセスでの実行
    with multiprocessing.Pool(processes=num_retry) as pool:
        result_iter = pool.imap_unordered(
            partial(
                run_one_seed, params, train_data, unit13, unit2, 
                period_log, thresholds, base_path
            ),
            list(range(num_retry))
        )
        
        success_found = False
        last_model = None
        
        for model, success, seed in result_iter:
            last_model = model
            if success:
                print(f"✅ 有効なモデルが見つかりました (seed={seed})")
                success_found = True
                break
            else:
                print(f"閾値を下回りませんでした (seed={seed})")
        
        pool.close()
        pool.join()
        
        if success_found:
            return model, True
        else:
            print(f"全ての初期値で閾値を下回るモデルは見つかりませんでした (unit13={unit13}, unit2={unit2})")
            print("----------------------------------------------------------------------------------------------------------------")
            return last_model, False


# =============================================================================
# ノード探索関連
# =============================================================================

def search_unit13_recursive(params, train_data, unit2, period_log, lower_bound, upper_bound, 
                            best_model, best_unit13, min_total_units, thresholds, base_path, searched_unit13):
    """
    unit13の再帰的二分探索
    
    入力:
        params: dict - ハイパーパラメータ
        train_data: ndarray - トレーニングデータ
        unit2: int - 中間層2のユニット数（固定）
        period_log: int - 期間番号
        lower_bound: int - 探索下限
        upper_bound: int - 探索上限
        best_model: CustomAutoencoder - 現在のベストモデル
        best_unit13: int - 現在のベストunit13
        min_total_units: int - 現在の最小合計ユニット数
        thresholds: dict - 特徴量ごとの基準値
        base_path: str - ベースパス
    出力:
        best_model: CustomAutoencoder - ベストモデル
        best_unit13: int - ベストunit13
        min_total_units: int - 最小合計ユニット数
    """
    # 変更後（打ち切り条件）
    mid_unit13 = (lower_bound + upper_bound) // 2

    if mid_unit13 in searched_unit13:
        print(f"[再帰探索終了] unit13={mid_unit13} は探索済み → 打ち切り")
        return best_model, best_unit13, min_total_units

    searched_unit13.add(mid_unit13)
    
    print("----------------------------------------------------------------------------------------------------------------")
    print(f"[再帰探索] unit13={mid_unit13}, unit2={unit2}")
    
    model, flag_low_threshold = try_init_point(
        params, train_data, mid_unit13, unit2, period_log, thresholds, base_path
    )
    total_units = mid_unit13 * 2 + unit2
    
    if flag_low_threshold:
        print(f"✅ 閾値を下回りました: unit13={mid_unit13}")
        if total_units < min_total_units:
            best_model = model
            best_unit13 = mid_unit13
            min_total_units = total_units
        
        # 再帰的に下限を更新
        return search_unit13_recursive(
            params, train_data, unit2, period_log,
            lower_bound, mid_unit13, best_model, best_unit13, min_total_units,
            thresholds, base_path, searched_unit13
        )
    else:
        # 再帰的に上限を更新
        return search_unit13_recursive(
            params, train_data, unit2, period_log,
            mid_unit13, upper_bound, best_model, best_unit13, min_total_units,
            thresholds, base_path, searched_unit13
        )


def search_optimal_units(params, train_data, period_log, thresholds, base_path):
    """
    メインノード探索ロジック
    
    入力:
        params: dict - ハイパーパラメータ
        train_data: ndarray - トレーニングデータ
        period_log: int - 期間番号
        thresholds: dict - 特徴量ごとの基準値
        base_path: str - ベースパス
    出力:
        best_model: CustomAutoencoder - 最適モデル
    """
    # 初期値の設定
    best_model = None
    best_unit_1_3 = None
    best_unit_2 = None
    min_total_units = float("inf")
    
    # ハイパーパラメータの取得
    units = params["unit"]
    first_unit13 = units * 2
    max_unit2 = units - 1
    
    # ユニット数を保持する変数
    previous_best_unit13 = None
    previous_unit2 = None
    
    # 終わりのフラグ
    end_flag = False
    
    # 探索開始
    for unit_2 in reversed(range(1, max_unit2 + 1)):
        print(f"\n===== unit2={unit_2} の探索開始 =====")
        
        if previous_best_unit13 is None:
            unit13 = first_unit13
        else:
            # 小数切り捨て
            unit13 = (previous_best_unit13 * 2 + previous_unit2 - unit_2) // 2
        
        print(f"unit2={unit_2} に対するunit13探索開始点: {unit13}")
        
        # 探索したユニット数を記録するセット
        searched_unit13 = set()
        
        # ノード探索に使用する変数
        lower = units + 1
        upper = None
        
        # 各unit13に対するupperを定義する
        if unit_2 == max_unit2:
            unit13 = first_unit13
            while True:
                print("----------------------------------------------------------------------")
                print(f"[unit{max_unit2}探索] unit13={unit13}, unit2={unit_2}, lower={lower}, upper={upper}")
                
                searched_unit13.add(unit13)
                
                # モデルの学習
                model, flag_low_threshold = try_init_point(
                    params, train_data, unit13, unit_2, period_log, thresholds, base_path
                )
                
                if flag_low_threshold:
                    # upperの代入
                    upper = unit13
                    total_units = unit13 * 2 + unit_2
                    min_total_units = total_units
                    best_model = model
                    best_unit_1_3 = unit13
                    best_unit_2 = unit_2
                    print(f"🎉 最良モデル更新: unit13={unit13}, unit2={unit_2}, 合計={total_units}")
                    unit13 = (upper + lower) // 2
                    break
                
                lower = unit13
                unit13 *= 2  # 見つかるまで倍々探索
        
        else:
            # unit2 < max_unit2 のとき：upper は過去のベスト構成から計算
            upper = ((best_unit_1_3 * 2 + best_unit_2) - unit_2) // 2
            unit13 = upper
        
        # unit13の探索
        while True:
            # 収束するか、入力ノード数を下回ったら終了
            if unit13 in searched_unit13 or unit13 <= units:
                print(f"探索が収束 or 範囲外（unit13={unit13}）。unit2={unit_2} の探索終了。")
                break
            
            searched_unit13.add(unit13)
            print(f"[探索の開始] unit13={unit13}, unit2={unit_2}")
            
            # モデルの学習
            model, flag_low_threshold = try_init_point(
                params, train_data, unit13, unit_2, period_log, thresholds, base_path
            )
            
            if not flag_low_threshold and len(searched_unit13) == 1 and unit_2 != max_unit2:
                # 初回の探索で閾値を下回らなかった場合、探索を終了
                print(f"unit2={unit_2} に対するunit13={unit13} でも閾値を下回らなかったので以降の探索を完全に終了します")
                
                if best_model is None:
                    print("❗ 最適モデルが見つかりませんでした。最後に探索したモデルを返します。")
                    return model
                
                end_flag = True
                break
            
            # 閾値を下回った場合、最適なユニット数を更新
            if flag_low_threshold:
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
                    params, train_data, unit_2, period_log,
                    lower, upper, best_model, unit13, min_total_units,
                    thresholds, base_path, searched_unit13
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
                    params, train_data, unit_2, period_log,
                    lower, upper, best_model, upper, min_total_units,
                    thresholds, base_path, searched_unit13
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