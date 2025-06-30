# custom_model.py

import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable


@register_keras_serializable()
class CustomAutoencoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data_for_monitoring = None
        self.reached_threshold = False

    def train_step(self, data):
        x = data[0]  # Autoencoderなのでx = y
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            # 各サンプルの誤差
            total_mse = tf.reduce_mean(tf.square(x_pred - x))  # スカラー
            total_rmase = tf.sqrt(total_mse)  # RMSE_total
            loss = total_rmase  # スカラー値としての損失

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    

class MaxReconstructionErrorEarlyStopping(keras.callbacks.Callback):
    def __init__(self, model, early_stopping_params, logger=None):
        super().__init__()
        self.target_model = model
        self.detail_log = early_stopping_params["detail_log"]
        self.threshold_avg = early_stopping_params["threshold_avg"]  # 0.0010
        self.threshold_max = early_stopping_params["threshold_max"]  # 0.0020
        self.log_counter = early_stopping_params["log_counter"]
        self.stopped_epoch = 0
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        x_all = self.target_model.train_data_for_monitoring
        x_pred = self.target_model(x_all, training=False)

        # 全データ全次元の MSE → スカラー
        total_mse = tf.reduce_mean(tf.square(x_pred - x_all))  # スカラー
        rmse_total = tf.sqrt(total_mse)  # スカラー → RMSE_total

        # 必要なら max_error は per-sample でも出せる（参考までに）
        mse_per_sample = tf.reduce_mean(tf.square(x_pred - x_all), axis=1)
        rmse_per_sample = tf.sqrt(mse_per_sample)


        max_error = tf.reduce_max(rmse_per_sample).numpy()
        avg_error = rmse_total.numpy()

        #倍数でログを出力
        if self.detail_log == True:
            if (epoch+1) % self.log_counter == 0:
                self.logger.info(f"Epoch {epoch + 1}: 平均誤差 = {avg_error:.6f}, 最大誤差 = {max_error:.6f}")

        if avg_error < self.threshold_avg and max_error < self.threshold_max:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.reached_threshold = True
            self.logger.info(f"モデルは条件を満たしたため学習を終了しました {epoch + 1}: 平均誤差={avg_error:.6f}, 最大誤差={max_error:.6f} < thresholds.")