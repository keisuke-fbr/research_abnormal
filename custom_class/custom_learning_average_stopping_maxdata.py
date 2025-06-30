# custom_model.py

import tensorflow as tf
from tensorflow import keras

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


class DualConditionEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, model, early_stopping_params):
        super().__init__()
        self.model_to_monitor = model
        self.max_error_threshold = early_stopping_params["max_error_threshold"]
        self.patience = early_stopping_params["patience"]
        self.min_delta = early_stopping_params["min_delta"]
        self.detail_log = early_stopping_params["detail_log"]
        self.wait = 0
        self.best_loss = float("inf")
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # 1. 平均的なloss（再構成誤差）を取得
        current_loss = logs.get("loss")
        
        # 2. 最大再構成誤差を取得
        max_error = float(self.model_to_monitor.compute_max_reconstruction_error())

        # 3. ログ出力（必要に応じて）
        if self.detail_log:
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch + 1}: Loss = {current_loss:.6f}, Max Error = {max_error:.6f}")

        # 4. lossが改善していれば記録更新
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        # 5. 条件両方満たしたら early stopping
        if self.wait >= self.patience and max_error < self.max_error_threshold:
            self.model.stop_training = True
            self.stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch + 1}: loss stopped improving and max error < {self.max_error_threshold} : max_error:{max_error}, average_error:{current_loss}")
