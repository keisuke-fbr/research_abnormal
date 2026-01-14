# custom_earlystopping.py
# カスタムオートエンコーダモデルと早期終了コールバック
# 修正版: 各ノード組の学習中にlog_counter間隔で即座にCSVへ書き込み

import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable
import numpy as np
import pandas as pd
import os

import config


@register_keras_serializable()
class CustomAutoencoder(keras.Model):
    """
    カスタムオートエンコーダモデル
    バッチ内で最大誤差のサンプル1つだけで勾配計算・重み更新を行う
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data_for_monitoring = None
        self.reached_threshold = False

    def train_step(self, data):
        """
        バッチ内最大誤差サンプルで勾配計算
        
        処理内容:
        1. バッチ内全サンプルの予測を計算
        2. 各サンプルのRMSEを計算
        3. 最大RMSEのサンプルを特定
        4. そのサンプルのみで勾配計算・重み更新
        """
        x = data[0]  # Autoencoderなのでx = y
        
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            
            # 各サンプルのRMSEを計算
            mse_per_sample = tf.reduce_mean(tf.square(x_pred - x), axis=1)
            rmse_per_sample = tf.sqrt(mse_per_sample)
            
            # 最大誤差のサンプルのインデックスを取得
            max_idx = tf.argmax(rmse_per_sample)
            
            # 最大誤差サンプルのみを抽出
            x_max = tf.gather(x, max_idx)
            x_max = tf.expand_dims(x_max, 0)  # バッチ次元を追加
            
            x_pred_max = tf.gather(x_pred, max_idx)
            x_pred_max = tf.expand_dims(x_pred_max, 0)
            
            # 最大誤差サンプルのRMSEを損失として使用
            mse_max = tf.reduce_mean(tf.square(x_pred_max - x_max))
            loss = tf.sqrt(mse_max)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}


class MaxReconstructionErrorEarlyStopping(keras.callbacks.Callback):
    """
    早期終了コールバック
    - 全特徴量が各基準値を下回ったら学習終了
    - log_counter間隔で学習ログを即座にCSVに書き込み（インクリメンタル保存）
    """
    
    def __init__(self, model, early_stopping_params, seed, period_log, unit13, unit2, base_path):
        """
        入力:
            model: CustomAutoencoder - 対象モデル
            early_stopping_params: dict - 早期終了パラメータ
                - detail_log: bool - 詳細ログ出力フラグ
                - thresholds: dict - 特徴量名→基準値のマッピング
                - log_counter: int - ログ記録エポック間隔
            seed: int - シード値
            period_log: int - 期間番号
            unit13: int - 中間層1, 3のユニット数
            unit2: int - 中間層2のユニット数
            base_path: str - ベースパス
        """
        super().__init__()
        self.target_model = model
        self.detail_log = early_stopping_params["detail_log"]
        self.thresholds = early_stopping_params["thresholds"]  # 特徴量ごとの基準値
        self.log_counter = early_stopping_params["log_counter"]
        # 改善判定の設定
        self.min_improve = early_stopping_params.get("min_improve", 0.005)
        self.no_improve_patience = early_stopping_params.get("no_improve_patience", 10000)
        self.stopped_epoch = 0
        
        # CSV保存用のパラメータ
        self.seed = seed
        self.period_log = period_log
        self.unit13 = unit13
        self.unit2 = unit2
        self.base_path = base_path
        
        # ファイルパスを事前に生成
        self.file_path = self._get_file_path()
        
        # ファイル初期化フラグ（ヘッダー書き込み済みかどうか）
        self.file_initialized = False
        
        # 早期終了用の状態（特徴ごと）
        self.best_min_error_per_feature = {}
        self.no_improve_count_per_feature = {}
    
    def _get_file_path(self):
        """
        CSVファイルパスを生成
        
        出力:
            file_path: str - CSVファイルパス
        """
        dir_path = os.path.join(
            self.base_path,
            f"learning_scores_seed{self.seed}",
            f"learning_scores_seed{self.seed}_period{self.period_log}"
        )
        os.makedirs(dir_path, exist_ok=True)
        
        file_name = f"learning_scores_seed{self.seed}_period{self.period_log}_unit13_{self.unit13}_unit2_{self.unit2}.csv"
        return os.path.join(dir_path, file_name)
    
    def _get_column_names(self):
        """
        CSVのカラム名リストを生成
        
        出力:
            columns: list - カラム名リスト
        """
        columns = ["epoch"]
        for col_name in config.columns_list:
            columns.append(f"{col_name}_avg")
            columns.append(f"{col_name}_max")
        return columns
    
    def _calculate_errors_per_feature(self, x_true, x_pred):
        """
        各特徴量ごとの誤差を計算
        
        入力:
            x_true: Tensor - 正解データ
            x_pred: Tensor - 予測データ
        出力:
            errors_avg: ndarray - 各特徴量の平均誤差
            errors_max: ndarray - 各特徴量の最大誤差
        """
        # 各データ、各特徴量の誤差を計算
        errors = tf.sqrt(tf.square(x_pred - x_true)).numpy()  # 形状: [N, 特徴数]
        
        # 各特徴量ごとの平均誤差
        errors_avg = np.mean(errors, axis=0)
        
        # 各特徴量ごとの最大誤差
        errors_max = np.max(errors, axis=0)
        
        return errors_avg, errors_max
    
    def _check_all_features_below_threshold(self, errors_per_feature):
        """
        全特徴量が基準値を下回っているかチェック
        
        入力:
            errors_per_feature: ndarray - 各特徴量の誤差
        出力:
            bool - 全特徴量が基準値を下回っていればTrue
        """
        for i, col_name in enumerate(config.columns_list):
            threshold = self.thresholds[col_name]
            if errors_per_feature[i] >= threshold:
                return False
        return True
    
    def _write_log_to_csv(self, epoch, errors_avg, errors_max):
        """
        学習ログを即座にCSVに書き込み（インクリメンタル保存）
        書き込み後は明示的にメモリを解放
        
        入力:
            epoch: int - 現在のエポック
            errors_avg: ndarray - 各特徴量の平均誤差
            errors_max: ndarray - 各特徴量の最大誤差
        """
        # 1行分のデータを作成
        row_data = {"epoch": epoch + 1}
        for i, col_name in enumerate(config.columns_list):
            row_data[f"{col_name}_avg"] = errors_avg[i]
            row_data[f"{col_name}_max"] = errors_max[i]
        
        # DataFrameに変換（1行）
        df_row = pd.DataFrame([row_data])
        
        # ファイルが初期化されていない場合はヘッダー付きで書き込み
        if not self.file_initialized:
            df_row.to_csv(self.file_path, index=False, mode='w')
            self.file_initialized = True
        else:
            # 追記モード（ヘッダーなし）
            df_row.to_csv(self.file_path, index=False, mode='a', header=False)
        
        # 明示的にメモリ解放
        del df_row
        del row_data
    
    def on_epoch_end(self, epoch, logs=None):
        """
        エポック終了時の処理
        - 誤差計算
        - ログ記録（log_counter間隔で即座にCSVに書き込み）
        - 早期終了判定
        """
        x_all = self.target_model.train_data_for_monitoring
        x_pred = self.target_model(x_all, training=False)
        
        # 各特徴量ごとの誤差を計算
        errors_avg, errors_max = self._calculate_errors_per_feature(x_all, x_pred)
        
        # log_counter間隔で即座にCSVに書き込み
        if (epoch + 1) % self.log_counter == 0:
            self._write_log_to_csv(epoch, errors_avg, errors_max)
        
        # 全特徴量が基準値を下回っているかチェック
        if self._check_all_features_below_threshold(errors_max):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.reached_threshold = True
            
            # 終了時もログを記録（log_counterの倍数でなくても）
            if (epoch + 1) % self.log_counter != 0:
                self._write_log_to_csv(epoch, errors_avg, errors_max)
            return
        
        # 基準値未達の特徴ごとに改善を判定
        for i, col_name in enumerate(config.columns_list):
            if errors_max[i] < self.thresholds[col_name]:
                continue
            
            current_error = float(errors_max[i])
            best_error = self.best_min_error_per_feature.get(col_name)
            
            if best_error is None:
                self.best_min_error_per_feature[col_name] = current_error
                self.no_improve_count_per_feature[col_name] = 0
                continue
            
            # 一定量改善したら更新、改善なしならカウントを進める
            if (best_error - current_error) >= self.min_improve:
                self.best_min_error_per_feature[col_name] = current_error
                self.no_improve_count_per_feature[col_name] = 0
            else:
                self.no_improve_count_per_feature[col_name] = (
                    self.no_improve_count_per_feature.get(col_name, 0) + 1
                )
            
            # 一定エポック連続で改善がなければ早期終了
            if self.no_improve_count_per_feature[col_name] >= self.no_improve_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(
                    f"早期終了: epoch={epoch + 1}, 特徴量={col_name}, "
                    f"現在値={current_error}, 過去最小値={best_error}, "
                    f"{self.no_improve_patience} エポック連続で {self.min_improve} 以上改善しませんでした。"
                )
                # 終了時もログを記録
                if (epoch + 1) % self.log_counter != 0:
                    self._write_log_to_csv(epoch, errors_avg, errors_max)
                break
    
    def on_train_end(self, logs=None):
        """
        学習終了時の処理
        - 最大エポック到達時、最終エポックがlog_counterの倍数でなければ書き込み
        """
        # 早期終了でない場合（最大エポックまで学習した場合）
        # 最終エポックのログがまだ書き込まれていなければ書き込む
        # ※ on_epoch_endで既に書き込み済みの場合は何もしない
        pass  # インクリメンタル書き込みのため、ここでの処理は不要
