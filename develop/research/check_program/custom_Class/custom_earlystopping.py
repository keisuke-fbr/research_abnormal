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
            mse_per_sample = tf.reduce_mean(tf.square(x_pred - x), axis=1)
            rmse_per_sample = tf.sqrt(mse_per_sample)
            loss = tf.reduce_mean(rmse_per_sample)
            

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    

class MaxReconstructionErrorEarlyStopping(keras.callbacks.Callback):
    def __init__(self, model, early_stopping_params, logger=None):
        super().__init__()
        self.target_model = model
        self.patience = early_stopping_params["patience"]
        self.min_delta = early_stopping_params["min_delta"]
        self.detail_log = early_stopping_params["detail_log"]
        self.patience_max = early_stopping_params["patience_max"]
        self.min_delta_max = early_stopping_params["min_delta_max"]
        self.confirm_datapoint = early_stopping_params["confirm_datapoint"]
        self.log_counter = early_stopping_params["log_counter"]

        self.stopped_epoch = 0
        self.wait = 0
        self.wait_max_error = 0
        self.best_loss = float("inf")
        self.best_max_error = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        x_all = self.target_model.train_data_for_monitoring
        x_pred = self.target_model(x_all, training=False)
        mse_per_sample = tf.reduce_mean(tf.square(x_pred - x_all), axis=1)
        rmse_per_sample = tf.sqrt(mse_per_sample)

        max_error = tf.reduce_max(rmse_per_sample).numpy()
        avg_error = tf.reduce_mean(rmse_per_sample).numpy()

        #倍数でログを出力
        if self.detail_log == True:
            if (epoch+1) % self.log_counter == 0:
                #ログに使用するある値よりも再構成誤差が大きいデータ点の個数を算出
                num_above_threshold = (rmse_per_sample > self.confirm_datapoint).numpy().sum()
                total_samples = len(rmse_per_sample)
                print(f"Epoch {epoch + 1}: 平均誤差 = {avg_error:.6f}, 最大誤差 = {max_error:.6f}")
                print(f"  Errors > {self.confirm_datapoint}: {num_above_threshold} / {total_samples} "
                  f"({100.0 * num_above_threshold / total_samples:.2f}%)")
        
        # 4. lossが改善していれば記録更新
        if avg_error < self.best_loss - self.min_delta:
            self.best_loss = avg_error
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
            print(f"モデルは条件を満たしたため学習を終了しました {epoch + 1}: 平均誤差={avg_error:.6f}, 最大誤差={max_error:.6f} < thresholds.")
        