# custom_earlystopping.py
# カスタムオートエンコーダモデルと早期終了コールバック

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
    - log_counter間隔で学習ログをCSVに保存
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
        self.stopped_epoch = 0
        
        # CSV保存用のパラメータ
        self.seed = seed
        self.period_log = period_log
        self.unit13 = unit13
        self.unit2 = unit2
        self.base_path = base_path
        
        # 学習ログを格納するリスト
        self.log_data = self._initialize_log_data()
    
    def _initialize_log_data(self):
        """
        学習ログデータの初期化
        
        出力:
            log_data: dict - エポックと各特徴量のavg/maxを格納する辞書
        """
        log_data = {"epoch": []}
        
        for col_name in config.columns_list:
            log_data[f"{col_name}_avg"] = []
            log_data[f"{col_name}_max"] = []
        
        return log_data
    
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
    
    def _record_log(self, epoch, errors_avg, errors_max):
        """
        学習ログを記録
        
        入力:
            epoch: int - 現在のエポック
            errors_avg: ndarray - 各特徴量の平均誤差
            errors_max: ndarray - 各特徴量の最大誤差
        """
        self.log_data["epoch"].append(epoch + 1)
        
        for i, col_name in enumerate(config.columns_list):
            self.log_data[f"{col_name}_avg"].append(errors_avg[i])
            self.log_data[f"{col_name}_max"].append(errors_max[i])
    
    def _save_learning_scores(self):
        """
        学習ログをCSVに保存
        """
        if len(self.log_data["epoch"]) == 0:
            return
        
        # DataFrameに変換
        df = pd.DataFrame(self.log_data)
        
        # ファイルパス生成
        dir_path = os.path.join(
            self.base_path,
            f"learning_scores_seed{self.seed}",
            f"learning_scores_seed{self.seed}_period{self.period_log}"
        )
        os.makedirs(dir_path, exist_ok=True)
        
        file_name = f"learning_scores_seed{self.seed}_period{self.period_log}_unit13_{self.unit13}_unit2_{self.unit2}.csv"
        file_path = os.path.join(dir_path, file_name)
        
        # 保存
        df.to_csv(file_path, index=False)
    
    def on_epoch_end(self, epoch, logs=None):
        """
        エポック終了時の処理
        - 誤差計算
        - ログ記録（log_counter間隔）
        - 早期終了判定
        """
        x_all = self.target_model.train_data_for_monitoring
        x_pred = self.target_model(x_all, training=False)
        
        # 各特徴量ごとの誤差を計算
        errors_avg, errors_max = self._calculate_errors_per_feature(x_all, x_pred)
        
        # log_counter間隔でログを記録
        if (epoch + 1) % self.log_counter == 0:
            self._record_log(epoch, errors_avg, errors_max)
        
        # 全特徴量が基準値を下回っているかチェック
        if self._check_all_features_below_threshold(errors_max):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.reached_threshold = True
            
            # 終了時もログを記録（log_counterの倍数でなくても）
            if (epoch + 1) % self.log_counter != 0:
                self._record_log(epoch, errors_avg, errors_max)
            
            # CSVに保存
            self._save_learning_scores()
    
    def on_train_end(self, logs=None):
        """
        学習終了時の処理
        - CSVに保存（まだ保存されていない場合）
        """
        # 早期終了でない場合（最大エポックまで学習した場合）もCSVに保存
        if not self.model.reached_threshold:
            self._save_learning_scores()
