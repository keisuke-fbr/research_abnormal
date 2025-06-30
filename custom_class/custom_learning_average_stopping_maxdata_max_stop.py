# custom_model.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

class CustomAutoencoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data_for_monitoring = None
    
    def compute_max_reconstruction_error(self):
            x_all = self.train_data_for_monitoring
            x_pred = self(x_all, training=False)
            mse_per_sample = tf.reduce_mean(tf.square(x_pred - x_all), axis=1)
            rmse_per_sample = tf.sqrt(mse_per_sample)
            return tf.reduce_max(rmse_per_sample)
    
    def compute_reconstruction_errors(self):
        """ 各データごとの再構成誤差を返す """
        x_all = self.train_data_for_monitoring
        x_pred = self(x_all, training=False)
        mse_per_sample = tf.reduce_mean(tf.square(x_pred - x_all), axis=1)
        rmse_per_sample = tf.sqrt(mse_per_sample)
        errors = tf.reshape(rmse_per_sample, (-1, 1))
        return rmse_per_sample.numpy()


class DualConditionEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, model, early_stopping_params):
        super().__init__()
        self.model_to_monitor = model
        self.patience = early_stopping_params["patience"]
        self.min_delta = early_stopping_params["min_delta"]
        self.detail_log = early_stopping_params["detail_log"]
        self.patience_max = early_stopping_params["patience_max"]
        self.min_delta_max = early_stopping_params["min_delta_max"]
        self.confirm_datapoint = early_stopping_params["confirm_datapoint"]

        self.stopped_epoch = 0
        self.wait = 0
        self.wait_max_error = 0
        self.best_loss = float("inf")
        self.best_max_error = float("inf")

        self.epoch_errors = []  # 各エポックの各データ再構成誤差を格納


    def on_epoch_end(self, epoch, logs=None):
        # 1. loss（再構成誤差）を取得
        errors = self.model_to_monitor.compute_reconstruction_errors()
        current_loss = float(errors.mean())  # 平均
        max_error = float(errors.max())      # 最大値


        # 3. ログ出力（必要に応じて）
        if self.detail_log:
            if (epoch+1) % 100 == 0:

                #ログに使用するある値よりも再構成誤差が大きいデータ点の個数を算出
                num_above_threshold = (errors > self.confirm_datapoint).sum()
                total_samples = len(errors)

                print(f"Epoch {epoch + 1}: Loss = {current_loss:.6f}, Max Error = {max_error:.6f}")
                print(f"  Errors > {self.confirm_datapoint}: {num_above_threshold} / {total_samples} "
                  f"({100.0 * num_above_threshold / total_samples:.2f}%)\n")
                
        # 4. lossが改善していれば記録更新
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        # 最大誤差に関するearly stopping判定
        if max_error < self.best_max_error - self.min_delta_max:
            self.best_max_error = max_error
            self.wait_max_error = 0
        else:
            self.wait_max_error += 1

        # 両方の条件が満たされたらearly stopping
        if self.wait >= self.patience and self.wait_max_error >= self.patience_max:
            self.model.stop_training = True
            self.stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch + 1}: loss and max error stopped improving.\n"
                  f"    Best loss: {self.best_loss}, Current loss: {current_loss}\n"
                  f"    Best max error: {self.best_max_error}, Current max error: {max_error}")
            

