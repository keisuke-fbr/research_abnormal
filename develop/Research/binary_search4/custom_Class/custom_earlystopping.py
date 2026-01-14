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
                - stagnation_epochs: int - 改善が見られない最大エポック数
                - stagnation_threshold: float - 改善とみなす最小変化量
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
        self.stopped_epoch = 0
        
        # 停滞検出用パラメータ
        self.stagnation_epochs = early_stopping_params.get("stagnation_epochs", 10000)
        self.stagnation_threshold = early_stopping_params.get("stagnation_threshold", 0.005)
        
        # 停滞検出用状態変数（特徴量ごと）
        # 各特徴量の「基準値」と「カウンタ」を保持
        self.stagnation_baseline = {}  # 特徴量名 → 基準値（最後に大きな改善があった時の最大誤差）
        self.stagnation_counter = {}   # 特徴量名 → カウンタ（改善がないエポック数）
        for col_name in config.columns_list:
            self.stagnation_baseline[col_name] = None  # 初回エポックで初期化
            self.stagnation_counter[col_name] = 0
        
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
        
        # 停止情報（親プロセスに返すため）
        self.stop_reason = None  # "threshold", "stagnation", "max_epochs"
        self.stagnant_feature = None  # 停滞終了時の原因特徴
    
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
    
    def _check_stagnation(self, errors_max):
        """
        基準値を満たしていない特徴の停滞をチェック
        
        入力:
            errors_max: ndarray - 各特徴量の最大誤差
        出力:
            is_stagnant: bool - 停滞している特徴が一つでもあればTrue
            stagnant_feature: str or None - 停滞している特徴名（停滞していなければNone）
        """
        for i, col_name in enumerate(config.columns_list):
            threshold = self.thresholds[col_name]
            current_error = errors_max[i]
            
            # 基準値を満たしている特徴はスキップ
            if current_error < threshold:
                continue
            
            # 初回は基準値を設定
            if self.stagnation_baseline[col_name] is None:
                self.stagnation_baseline[col_name] = current_error
                self.stagnation_counter[col_name] = 0
                continue
            
            # 基準値からの改善量を計算
            improvement = self.stagnation_baseline[col_name] - current_error
            
            # 改善閾値以上の改善があった場合
            if improvement >= self.stagnation_threshold:
                # 基準値を更新し、カウンタをリセット
                self.stagnation_baseline[col_name] = current_error
                self.stagnation_counter[col_name] = 0
            else:
                # 改善が不十分な場合、カウンタを増加
                self.stagnation_counter[col_name] += 1
                
                # 停滞エポック数に達したら終了
                if self.stagnation_counter[col_name] >= self.stagnation_epochs:
                    return True, col_name
        
        return False, None

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
    
    def get_stop_info(self):
        """
        停止情報を取得（親プロセスに返すため）
        
        出力:
            stop_info: dict - 停止情報
        """
        return {
            "stop_reason": self.stop_reason if self.stop_reason else "max_epochs",
            "stopped_epoch": self.stopped_epoch + 1 if self.stopped_epoch > 0 else None,
            "stagnant_feature": self.stagnant_feature,
            "stagnation_epochs": self.stagnation_epochs,
            "stagnation_threshold": self.stagnation_threshold
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """
        エポック終了時の処理
        - 誤差計算
        - ログ記録（log_counter間隔で即座にCSVに書き込み）
        - 早期終了判定（基準値達成 or 停滞検出）
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
            self.stop_reason = "threshold"
            
            # 終了時もログを記録（log_counterの倍数でなくても）
            if (epoch + 1) % self.log_counter != 0:
                self._write_log_to_csv(epoch, errors_avg, errors_max)
            return
        
        # 停滞チェック（基準値を満たしていない特徴の改善が停滞していないか）
        is_stagnant, stagnant_feature = self._check_stagnation(errors_max)
        if is_stagnant:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.reached_threshold = False  # 基準値未達で終了
            self.stop_reason = "stagnation"
            self.stagnant_feature = stagnant_feature
            
            # 終了時もログを記録
            if (epoch + 1) % self.log_counter != 0:
                self._write_log_to_csv(epoch, errors_avg, errors_max)
    
    
    def on_train_end(self, logs=None):
        """
        学習終了時の処理
        - 最大エポック到達時、最終エポックがlog_counterの倍数でなければ書き込み
        """
        # 早期終了でない場合（最大エポックまで学習した場合）
        # 最終エポックのログがまだ書き込まれていなければ書き込む
        # ※ on_epoch_endで既に書き込み済みの場合は何もしない
        pass  # インクリメンタル書き込みのため、ここでの処理は不要