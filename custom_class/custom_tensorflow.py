# custom_model.py

import tensorflow as tf
from tensorflow import keras

class CustomAutoencoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data_for_monitoring = None

    def train_step(self, data):
        x = data[0]  # Autoencoderなのでx = y
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            errors = tf.reduce_mean(tf.square(x_pred - x), axis=1)  # 各サンプルの誤差

            # 最大誤差を持つサンプルを取得
            max_idx = tf.argmax(errors)
            x_single = tf.expand_dims(x[max_idx], axis=0)

            # 対象サンプルで再計算
            x_pred_single = self(x_single, training=True)
            loss = tf.sqrt(tf.reduce_mean(tf.square(x_pred_single - x_single)))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
    
    def compute_max_reconstruction_error(self):
        x_all = self.train_data_for_monitoring
        x_pred = self(x_all, training=False)
        errors = tf.reduce_mean(tf.square(x_pred - x_all), axis=1)
        return tf.reduce_max(errors)
    

class MaxReconstructionErrorEarlyStopping(keras.callbacks.Callback):
    def __init__(self, model, early_stopping_params):
        super().__init__()
        self.target_model = model
        self.patience = early_stopping_params["patience"]
        self.min_delta = early_stopping_params["min_delta"]
        self.detail_log = early_stopping_params["detail_log"]
        self.wait = 0
        self.best = float("inf")
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        max_error = float(self.target_model.compute_max_reconstruction_error())

        #100の倍数でログを出力
        if self.detail_log == True:
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch + 1}: Max reconstruction error = {max_error:.5f}")

        if max_error < self.best - self.min_delta:
            self.best = max_error
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"Stopped at epoch {epoch + 1} due to early stopping (max error did not improve).")
