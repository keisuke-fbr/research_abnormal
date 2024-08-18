# 必要ライブラリのインストール
import sys
import pandas as pd
import numpy as np
from sklearn import svm
import scipy.stats
import matplotlib.pyplot as plt
import csv
import re
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn import preprocessing
import datetime
from dfply import *
import random
import gc
import shap
plt.style.use('ggplot')
# pd.options.display.float_format = '{:.2f}'.format

# %% [markdown]
# 生データの読み込み

# %% [markdown]
# ECDISデータの読み込み

# %%
raw_data_ECDIS = pd.read_csv('../data/ECDIS__FMD-3x00.csv', header=0, delimiter=',',dtype = {"equipment_label_no":"object","f_shipno":"object"})

raw_data_ECDIS = raw_data_ECDIS >> mutate(id = X.f_shipno+"-"+X.equipment_label_no,
                         measurement_ymd = X.measurement_date.str[:10],
                         measurement_ymd_h = X.measurement_date.str[11:13]) >> mutate(
    measurement_ymd_hms = X.measurement_ymd+" "+X.measurement_ymd_h+":00:00")

# %% [markdown]
# Radarデータの読み込み

# %%
raw_data_Radar= pd.read_csv('../data/Radar__FAR-3xx0.csv', header=0, delimiter=',',dtype = {"equipment_label_no":"object","f_shipno":"object"})
raw_data_Radar = raw_data_Radar>> mutate(id = X.f_shipno+"-"+X.equipment_label_no,
                         measurement_ymd = X.measurement_date.str[:10],
                         measurement_ymd_h = X.measurement_date.str[11:13]) >> mutate(
    measurement_ymd_hms = X.measurement_ymd+" "+X.measurement_ymd_h+":00:00")

# %% [markdown]
# 必要なカラムを抽出、および半角処理

# %%
raw_data_ECDIS_select = raw_data_ECDIS[["id","equipment_label_no","f_shipno","measurement_date"
                                        ,"processor_unit_units_hardware_info__serial_number_cpu_bd","monitor1_units__unit","monitor2_units__unit"
                                        ,"measurement_ymd","measurement_ymd_h","measurement_ymd_hms"
                                        ,"monitor1_units_status_main__temp","monitor1_units_status_main__fan1"
                                        ,"monitor1_units_status_main__fan2","monitor1_units_status_main__fan3"
                                        ,"monitor2_units_status_main__temp","monitor2_units_status_main__fan1"
                                        ,"monitor2_units_status_main__fan2","monitor2_units_status_main__fan3","processor_unit_units_status_cpu_board__cpu_fan"
                                        ,"processor_unit_units_status_cpu_board__cpu_bd_temp","processor_unit_units_status_cpu_board__cpu_core_temp"
                                        ,"processor_unit_units_status_cpu_board__gpu_core_temp","processor_unit_units_status_cpu_board__remote1_temp"
                                        ,"processor_unit_units_status_cpu_board__remote2_temp","processor_unit_units_status_cpu_board__cpu_core_vol"
                                        ,"processor_unit_units_status_cpu_board__cpu_bd_vbat","processor_unit_units_status_cpu_board__cpu_bd_p3_3v"
                                        ,"processor_unit_units_status_cpu_board__cpu_bd_p5v","processor_unit_units_status_cpu_board__cpu_bd_p12v"
                                        ,"processor_unit_units_status_cpu_board__cpu_bd_fan1","processor_unit_units_status_cpu_board__cpu_bd_fan2"
                                        ,"processor_unit_units_status_boot_device__wearout_ind"]]


data_ECDIS=raw_data_ECDIS_select.rename(columns={"processor_unit_units_hardware_info__serial_number_cpu_bd":"processor_unit_units_hardware_info_serial_number_cpu_bd"
                                                 ,"monitor1_units__unit":"monitor1_units_unit"
                                                 ,"monitor2_units__unit":"monitor2_units_unit"
                                                 ,"monitor1_units_status_main__temp":"monitor1_units_status_main_temp"
                                                 ,"monitor1_units_status_main__fan1":"monitor1_units_status_main_fan1"
                                                 ,"monitor1_units_status_main__fan2":"monitor1_units_status_main_fan2"
                                                 ,"monitor1_units_status_main__fan3":"monitor1_units_status_main_fan3"
                                                 ,"monitor2_units_status_main__temp":"monitor2_units_status_main_temp"
                                                 ,"monitor2_units_status_main__fan1":"monitor2_units_status_main_fan1"
                                                 ,"monitor2_units_status_main__fan2":"monitor2_units_status_main_fan2"
                                                 ,"monitor2_units_status_main__fan3":"monitor2_units_status_main_fan3"
                                                 ,"processor_unit_units_status_cpu_board__cpu_fan":"processor_unit_units_status_cpu_board_cpu_fan"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_temp":"processor_unit_units_status_cpu_board_cpu_bd_temp"
                                                 ,"processor_unit_units_status_cpu_board__cpu_core_temp":"processor_unit_units_status_cpu_board_cpu_core_temp"
                                                 ,"processor_unit_units_status_cpu_board__gpu_core_temp":"processor_unit_units_status_cpu_board_gpu_core_temp"
                                                 ,"processor_unit_units_status_cpu_board__remote1_temp":"processor_unit_units_status_cpu_board_remote1_temp"
                                                 ,"processor_unit_units_status_cpu_board__remote2_temp":"processor_unit_units_status_cpu_board_remote2_temp"
                                                 ,"processor_unit_units_status_cpu_board__cpu_core_vol":"processor_unit_units_status_cpu_board_cpu_core_vol"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_vbat":"processor_unit_units_status_cpu_board_cpu_bd_vbat"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_p3_3v":"processor_unit_units_status_cpu_board_cpu_bd_p3_3v"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_p5v":"processor_unit_units_status_cpu_board_cpu_bd_p5v"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_p12v":"processor_unit_units_status_cpu_board_cpu_bd_p12v"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_fan1":"processor_unit_units_status_cpu_board_cpu_bd_fan1"
                                                 ,"processor_unit_units_status_cpu_board__cpu_bd_fan2":"processor_unit_units_status_cpu_board_cpu_bd_fan2"
                                                 ,"processor_unit_units_status_boot_device__wearout_ind":"processor_unit_units_status_boot_device_wearout_ind"
                                                })

# %%
data_Radar = raw_data_Radar[["id","equipment_label_no","f_shipno","measurement_date",
                             "processor_unit_units_hardware_info_serial_number_cpu_bd","monitor1_units_unit","monitor2_units_unit",
                             "measurement_ymd","measurement_ymd_h","measurement_ymd_hms",
                             "monitor1_units_status_main_temp","monitor1_units_status_main_fan1",
                            "monitor1_units_status_main_fan2","monitor1_units_status_main_fan3"
                             ,"monitor2_units_status_main_temp"
                             ,"monitor2_units_status_main_fan1","monitor2_units_status_main_fan2"
                             ,"monitor2_units_status_main_fan3","processor_unit_units_status_cpu_board_cpu_fan",
                            "processor_unit_units_status_cpu_board_cpu_bd_temp","processor_unit_units_status_cpu_board_cpu_core_temp",
                            "processor_unit_units_status_cpu_board_gpu_core_temp","processor_unit_units_status_cpu_board_remote1_temp",
                            "processor_unit_units_status_cpu_board_remote2_temp","processor_unit_units_status_cpu_board_cpu_core_vol",
                            "processor_unit_units_status_cpu_board_cpu_bd_vbat","processor_unit_units_status_cpu_board_cpu_bd_p3_3v",
                            "processor_unit_units_status_cpu_board_cpu_bd_p5v","processor_unit_units_status_cpu_board_cpu_bd_p12v",
                            "processor_unit_units_status_cpu_board_cpu_bd_fan1","processor_unit_units_status_cpu_board_cpu_bd_fan2",
                            "processor_unit_units_status_storage_device_wearout_ind"]]

# %% [markdown]
# カラムの説明

# %% [markdown]
# ここでデータのカラム値は以下の通りである。
# カラム名	説明
# 
# id	                                                船舶番号 (f_shipno) と装置ラベル番号 (equipment_label_no) を結合したもの（ユニークID）
# 
# equipment_label_no	                                装置ラベル番号
# 
# f_shipno	                                        船舶番号
# 
# measurement_date	                                測定日時
# 
# processor_unit_units_hardware_info_serial_number_cpu_bd	CPUボードのシリアル番号
# 
# monitor1_units_unit	                                モニター1のユニット情報
# 
# monitor2_units_unit	                                モニター2のユニット情報
# 
# measurement_ymd	                                    測定日の年月日部分（YYYY-MM-DD）
# 
# measurement_ymd_h	                                測定日の時間部分（HH）
# 
# measurement_ymd_hms	                                測定日の詳細な日時（YYYY-MM-DD HH:00:00）
# 
# monitor1_units_status_main_temp	                    モニター1のメインステータスの温度
# 
# monitor1_units_status_main_fan1	                    モニター1のメインステータスのファン1
# 
# monitor1_units_status_main_fan2	                    モニター1のメインステータスのファン2
# 
# monitor1_units_status_main_fan3	                    モニター1のメインステータスのファン3
# 
# monitor2_units_status_main_temp	                    モニター2のメインステータスの温度
# 
# monitor2_units_status_main_fan1	                    モニター2のメインステータスのファン1
# 
# monitor2_units_status_main_fan2	                    モニター2のメインステータスのファン2
# 
# monitor2_units_status_main_fan3	                    モニター2のメインステータスのファン3
# 
# processor_unit_units_status_cpu_board_cpu_fan	    CPUボードのCPUファンステータス
# 
# processor_unit_units_status_cpu_board_cpu_bd_temp	CPUボードの温度
# 
# processor_unit_units_status_cpu_board_cpu_core_temp	CPUコアの温度
# 
# processor_unit_units_status_cpu_board_gpu_core_temp	GPUコアの温度
# 
# processor_unit_units_status_cpu_board_remote1_temp	リモート1の温度
# 
# processor_unit_units_status_cpu_board_remote2_temp	リモート2の温度
# 
# processor_unit_units_status_cpu_board_cpu_core_vol	CPUコアの電圧
# 
# processor_unit_units_status_cpu_board_cpu_bd_vbat	CPUボードのバッテリ電圧
# 
# processor_unit_units_status_cpu_board_cpu_bd_p3_3v	CPUボードの3.3V電源
# 
# processor_unit_units_status_cpu_board_cpu_bd_p5v	CPUボードの5V電源
# 
# processor_unit_units_status_cpu_board_cpu_bd_p12v	CPUボードの12V電源
# 
# processor_unit_units_status_cpu_board_cpu_bd_fan1	CPUボードのファン1
# 
# processor_unit_units_status_cpu_board_cpu_bd_fan2	CPUボードのファン2
# 
# processor_unit_units_status_boot_device_wearout_ind	起動デバイスの摩耗インデックス

# %% [markdown]
# データの結合

# %%
con=pd.concat([data_Radar, data_ECDIS])
data_con=con

# %% [markdown]
# キャッシュの削除

# %%
del raw_data_ECDIS
del raw_data_Radar
gc.collect()

# %% [markdown]
# データの確認

# %% [markdown]
# なお、「f_shipno」は船舶番号、「equipment_label_no」は装置ラベル番号、「id」は船舶番号と装置ラベル番号を結合したもので装置を特定する。

# %%
print("ECDIS&Radar")
print("idユニーク数                :",len(data_con["id"].unique()))
print("f_shipnoユニーク数          :",len(data_con["f_shipno"].unique()))
print("equipment_label_noユニーク数:",len(data_con["equipment_label_no"].unique()),"\n")

# %%
# raw_data.shape
print("行数: "+str(data_con.shape[0]))
print("列数: "+str(data_con.shape[1]))
#データ期間
print("データ開始日時: "+str(data_con["measurement_date"].min()))
print("データ終了日時: "+str(data_con["measurement_date"].max()))
print("f_shipno(ユニーク数): "+str(data_con["f_shipno"].nunique()))
print("equipment_label_no(ユニーク数): "+str(data_con["equipment_label_no"].nunique()))

# %% [markdown]
# 日付のフォーマットの変更

# %%
data_con["measurement_ymd_hms"] = pd.to_datetime(data_con["measurement_ymd_hms"], format='%Y-%m-%d %H:%M:%S')

# %% [markdown]
# データの加工
# 
# ・一時間に一個のデータしか用いない（最初のデータ）
# 
# ・一日取得データ数１５件以上のみ使用

# %%
# 1時間に1データに変更
data_con["measurement_date"] = pd.to_datetime(data_con["measurement_date"], format='%Y-%m-%d %H:%M:%S')

data_con["order"] = data_con.groupby(["id", "measurement_ymd", "measurement_ymd_h"])["measurement_date"].rank()

modified_data = data_con[data_con["order"]==1]

# %%
check_day_cnt = modified_data.groupby(["id", "measurement_ymd"]).agg({"measurement_date":"nunique"}).assign(
    min_measurement_date = modified_data.groupby(['id', "measurement_ymd"]).agg({"measurement_date":"min"}),
    max_measurement_date = modified_data.groupby(['id', "measurement_ymd"]).agg({"measurement_date":"max"})).reset_index()

check_day_cnt["id_date"] = check_day_cnt["id"]+"-"+check_day_cnt["measurement_ymd"]

check_day_cnt = check_day_cnt.rename(columns={"measurement_date":"cnt"})

target_id_date = check_day_cnt[check_day_cnt["cnt"] >= 15]["id_date"]

# %% [markdown]
# 条件２の適用

# %%
modified_data["tag"] = modified_data["id"]+"-"+modified_data["measurement_ymd"]
# modified_data.head()

final_data = modified_data[modified_data["tag"].isin(target_id_date)]
final_data.shape

# %%


# %% [markdown]
# ここまでが生データに対する処理である
# 
# データは：final_dataである

# %% [markdown]
# 以下では、発見されている異常データのインスタンスを含むようなデータを抽出する

# %% [markdown]
# cpuのシリアル番号によって分類されるADP555を抜き出す

# %%
(final_data["processor_unit_units_hardware_info_serial_number_cpu_bd"].str[1:3]).unique()

# %%
data_ADP555 = final_data[((final_data["processor_unit_units_hardware_info_serial_number_cpu_bd"].str[1:3])
                          .isin(["16","15","14","13"]))]


# %% [markdown]
# データの確認

# %%
data_ADP555["measurement_date"].describe()

# %% [markdown]
# 異常データの確認
# 
# 今回確認されている異常データは
# 
# id = "9748019T-323"で確認された
# 
# 2028/08/15付近の
# 
# processor_unit_units_status_cpu_board_cpu_bd_fan2
# 
# から確認されるものである

# %%
#期間始め
s_day = pd.to_datetime("2018/06/01 00:00:00", format='%Y/%m/%d %H:%M:%S')
#期間終わり
e_day=pd.to_datetime("2018/10/1 00:00:00", format='%Y/%m/%d %H:%M:%S')

#特定データの抜き出し
ab_date=data_ADP555[data_ADP555["id"]=="9748019T-325"]

test_select_df = ab_date[(ab_date["measurement_date"]>=s_day)&(ab_date["measurement_date"]<e_day)]


time = pd.to_datetime(test_select_df["measurement_date"], format='%Y/%m/%d %H:%M:%S')


fig = plt.figure(figsize=(30,7))
ax1 = fig.add_subplot(1,1,1)
ax1.scatter(time,test_select_df["processor_unit_units_status_cpu_board_cpu_bd_fan2"], color='blue')
ax1.set_xlim(time.min(),time.max())
ax1.legend()

# %% [markdown]
# 以下の実装では、上記の異常データを発見するとともに、異常の原因をその特徴量であると発見し、さらに予兆を検知できるようなモデルを作成することを目標とする

# %% [markdown]
# 異常を含むidのデータセットを作成
# 
# すなわち特定の機器に対してモデルを作成する

# %%
data_model=data_ADP555[data_ADP555["id"]=="9748019T-325"]
colums_list=["processor_unit_units_status_cpu_board_cpu_fan","processor_unit_units_status_cpu_board_cpu_bd_fan1"
             ,"processor_unit_units_status_cpu_board_cpu_bd_fan2","processor_unit_units_status_cpu_board_cpu_bd_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_temp","processor_unit_units_status_cpu_board_gpu_core_temp"
             ,"processor_unit_units_status_cpu_board_cpu_core_vol","processor_unit_units_status_cpu_board_cpu_bd_vbat"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p3_3v","processor_unit_units_status_cpu_board_cpu_bd_p5v"
             ,"processor_unit_units_status_cpu_board_cpu_bd_p12v"]

# %% [markdown]
# 異常検知に用いられるカラムは以下の通り
# 
# カラム名	説明
# 
# processor_unit_units_status_cpu_board_cpu_fan	CPUボードのCPUファンのステータス
# 
# processor_unit_units_status_cpu_board_cpu_bd_fan1	CPUボードのファン1のステータス
# 
# processor_unit_units_status_cpu_board_cpu_bd_fan2	CPUボードのファン2のステータス
# 
# processor_unit_units_status_cpu_board_cpu_bd_temp	CPUボードの温度
# 
# processor_unit_units_status_cpu_board_cpu_core_temp	CPUコアの温度
# 
# processor_unit_units_status_cpu_board_gpu_core_temp	GPUコアの温度
# 
# processor_unit_units_status_cpu_board_cpu_core_vol	CPUコアの電圧
# 
# processor_unit_units_status_cpu_board_cpu_bd_vbat	CPUボードのバッテリ電圧
# 
# processor_unit_units_status_cpu_board_cpu_bd_p3_3v	CPUボードの3.3V電源
# 
# processor_unit_units_status_cpu_board_cpu_bd_p5v	CPUボードの5V電源
# 
# processor_unit_units_status_cpu_board_cpu_bd_p12v	CPUボードの12V電源
# 
# 

# %%
data_model


# %% [markdown]
# 実験条件の整理
# 
# 今、データは39か月存在している。
# 
# １か月のデータを直近２年のデータを使用して予測を行うので
# 合計15回実験を行う

# %% [markdown]
# 以下、autoencoderの実装を行う

# %% [markdown]
# まずはデータの準備から

# %%
#データの標準化（０～１にする）
scaler = preprocessing.MinMaxScaler()
data_model[colums_list] = scaler.fit_transform(data_model[colums_list])
data_model[colums_list] += 0.00001

# 保存先
ans_data=pd.DataFrame()

#実装に必要なカラムと取得日時のカラムの作成
data_ex = data_model[colums_list+["measurement_date"]]

#欠損値処理（欠損値があった場合は落とす）
data_ex=data_ex.dropna(how='any')
data_ex=data_ex.reset_index(drop = True)
data_ex["measurement_date"]=pd.to_datetime(data_ex["measurement_date"], format='%Y/%m/%d %H:%M:%S')


#初期データの抜き出し及び、繰り返し処理の準備
data_trainestart = data_ex["measurement_date"][0]
data_trainestart_year = data_trainestart.year
data_trainestart_month = data_trainestart.month
data_trainestart = str(data_trainestart_year) + "/" + str(data_trainestart_month) + "/01 00:00:00"

data_trainend_year = data_trainestart_year + 2
data_trainend_month = data_trainestart_month
data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

data_teststart = data_trainend
if data_trainend_month == 12:
    data_testend_year = data_trainend_year + 1
    data_testend_month = 1
else:
    data_testend_year = data_trainend_year
    data_testend_month = data_trainend_month + 1

data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

print("初期データ(train)日付はじまり: "+ str(data_trainestart))
print("初期データ(train)日付終わり: "+str(data_trainend))

print("初期データ(test)日付はじまり: "+ str(data_teststart))
print("初期データ(test)日付終わり: "+str(data_testend))



#繰り返し回数
num = 15


# %%
data_ex.columns

# %% [markdown]
# ここからモデル作成に入る

# %%
#必要ライブラリのインストール

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setuptools
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# %% [markdown]
# モデルの準備
# 
# 今回はkerasのsequentialを用いてモデルを作成する

# %% [markdown]
# ハイパーパラメータについて
# 
# １，中間層のユニット数
# 
# ２，中間層の活性化関数
# 
# ３，出力層の活性化関数
# 
# ４，モデルの最適化手法
# 
# ５，モデルの最小化関数
# 
# ６，異常スコアの関数
# 
# ７，閾値
# 
# ８，エポック数

# %% [markdown]
# 自作損失関数の定義(モデルの最小化関数)

# %%
# カスタム損失関数の定義
def custom_loss(delta):
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        small_error_loss = tf.square(error)
        large_error_loss = error
        return tf.where(error < delta, large_error_loss, small_error_loss)
    return loss

# %% [markdown]
# モデル作成の関数化

# %%
#モデル作成
def model_autoencoder():
    # モデルを作成
    model = keras.Sequential(name="autoencoder")

    #入力層の情報
    input_unit = 11

    #中間層の情報
    middle_unit = 8

    #出力層の情報
    output_unit = 11

    #層の追加
    #中間層の作成
    model.add(layers.Dense(units=middle_unit, activation="relu" ,input_shape=(input_unit,)))

    #出力層の作成
    model.add(layers.Dense(units=output_unit, activation="sigmoid"))


    #モデルのコンパイル
    model.compile(optimizer="adam", loss=custom_loss(delta=0.5))

    return model

# %% [markdown]
# 閾値の設定

# %%
#閾値の設定
threshold = 0.8

# %% [markdown]
# 異常スコア算出関数

# %%
#二乗誤差を取るだけ
#大きいほど元のデータと離れているので異常
def abnomalScores(originalDF,reduceDF):
  loss = np.sum((np.array(originalDF)-np.array(reduceDF))**2,axis=1)
  loss = pd.Series(data=loss,index=originalDF.index)
  loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
  return loss

# %% [markdown]
# 一回目のモデルの実装部分（二度目の学習データに異常と判定されたデータを除く）

# %%
#結果（日付と異常値スコアを含む配列の準備）
# 空のデータフレームを作成
results_df = pd.DataFrame(columns=["measurement_date", "anomaly_score"])
abnormal_df = pd.DataFrame(columns=["measurement_date", "anomaly_score"])

#正常データを格納する配列
normal_df = pd.DataFrame(columns=["measurement_date"] + colums_list + ["anomaly_score"])

#初期値の変更回数
num_initializations = 5


#初期データの抜き出し及び、繰り返し処理の準備
data_trainestart = data_ex["measurement_date"][0]
data_trainestart_year = data_trainestart.year
data_trainestart_month = data_trainestart.month
data_trainestart = str(data_trainestart_year) + "/" + str(data_trainestart_month) + "/01 00:00:00"

data_trainend_year = data_trainestart_year + 2
data_trainend_month = data_trainestart_month
data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

data_teststart = data_trainend
if data_trainend_month == 12:
    data_testend_year = data_trainend_year + 1
    data_testend_month = 1
else:
    data_testend_year = data_trainend_year
    data_testend_month = data_trainend_month + 1

data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

# 初めの2年間のデータをそのままnormal_dfに格納
initial_normal_data = data_ex[(data_ex["measurement_date"] >= "2015/12/01 00:00:00") & (data_ex["measurement_date"] < "2017/12/01 00:00:00")]
normal_df = pd.concat([normal_df, initial_normal_data], ignore_index=True)


#モデル実装部分
for i in range(num):
    print("============================================")
    print(str(i+1) + "回目のモデル")
    #i回目のデータの抜き出し

    #トレーニングデータの抜き出し
    train_data = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainestart)]
    train_data = train_data[colums_list]
    train_data = train_data.values


    #テストデータの抜き出し
    #テストデータはdata_trainendの一か月である
    test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
    test_data_origin = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
    test_data = test_data[colums_list]
    test_data = test_data.values

    #該当データ期間
    print("はじまり(train)" + str(data_trainestart))
    print("終わり(train)" + str(data_trainend))
    print("はじまり(test)" + str(data_teststart))
    print("終わり(test)" + str(data_testend))


    # 初期化ごとの最良のモデルを保持する変数
    best_model = None
    best_loss = np.inf

    #初期点変更による複数回モデルの作成
    for k in range(num_initializations):
        print(f"Initialization {k+1}/{num_initializations}")

        # モデルの作成
        model = model_autoencoder()

        #モデルの学習
        history = model.fit(train_data, train_data,
                epochs=10,
                shuffle=True,
                validation_data=(train_data, train_data))
        
        # 最良の損失値を保持し、最良のモデルを選ぶ
        val_loss = history.history['val_loss'][-1]
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
    
    #結果
    result_data = best_model.predict(test_data)

    #pandas形式へ変換
    test_data = pd.DataFrame(test_data,columns=colums_list)
    print(len(test_data))
    result_data = pd.DataFrame(result_data,columns=colums_list)

    #異常値の算出
    abnormal_value = abnomalScores(test_data,result_data)



    #結果の格納
    temp_df = pd.DataFrame({
        "measurement_date": test_data_origin["measurement_date"].values,
        "anomaly_score": abnormal_value
    })

    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    #異常値、正常値データの格納
    for j in range(len(abnormal_value)):
        if abnormal_value[j] > threshold:
            # 1行のデータフレームを作成
            tempab_df = pd.DataFrame({
                "measurement_date": [test_data_origin["measurement_date"].iloc[j]],
                "anomaly_score": [abnormal_value[j]]
            })
    
            # 異常値のみを格納するデータフレームに追加
            abnormal_df = pd.concat([abnormal_df, tempab_df], ignore_index=True)

            
        else:
            # 正常値のみを格納するデータフレームに追加
            temp_normal_df = pd.DataFrame({
                "measurement_date": [test_data_origin["measurement_date"].iloc[j]],
                "anomaly_score": [abnormal_value[j]]
            })
            for col in colums_list:
                temp_normal_df[col] = test_data_origin[col].iloc[j]

            
            normal_df = pd.concat([normal_df, temp_normal_df], ignore_index=True)


    
    


    #日付の更新
    if data_trainestart_month == 12:
        data_trainestart_year += 1
        data_trainestart_month = 1
    else:
        data_trainestart_month += 1
    data_trainestart = str(data_trainestart_year) + "/" + str(data_trainestart_month) + "/01 00:00:00"        
    
    data_trainend_year = data_trainestart_year + 2
    data_trainend_month = data_trainestart_month
    data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

    data_teststart = data_trainend
    if data_trainend_month == 12:
        data_testend_year = data_trainend_year + 1
        data_testend_month = 1
    else:
        data_testend_year = data_trainend_year
        data_testend_month = data_trainend_month + 1

    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

print(results_df)
print(abnormal_df)


    

# %% [markdown]
# 一回目のデータのプロット

# %%
#散布図の作成

#異常スコアの出力
#図の大きさを指定
fig = plt.figure(figsize=(60,100))

#はじめのサブプロットの作成（異常スコア）
ax1 = fig.add_subplot(14,1,1)
ax1.set_title("autoencoder_score")
#正常スコアと異常スコア
ax1.scatter(results_df["measurement_date"], results_df["anomaly_score"], c='blue', marker='o', edgecolor='k')
ax1.scatter(abnormal_df["measurement_date"], abnormal_df["anomaly_score"], c='red', marker='o', edgecolor='k')
ax1.set_ylabel('Abnormality')
# ラベルの設定
ax1.set_ylabel('Abnormality')
# y=0.8の位置に赤い横線を引く
ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2)


#正常スコアのみのプロット
ax2 = fig.add_subplot(14,1,2)
ax2.set_title("autoencoder_score_normal")
# 縦軸の範囲を0から1に設定
ax2.set_ylim(0, 1)
ax2.scatter(normal_df["measurement_date"], normal_df["anomaly_score"], c='blue', marker='o', edgecolor='k')
ax2.set_ylabel('normality')
# ラベルの設定
ax2.set_ylabel('normality')
# y=0.8の位置に赤い横線を引く
ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2)


# 元の特徴量の時系列ごとのデータ
for i, column in enumerate (colums_list):
    ax = fig.add_subplot(13,1,i+3)
    ax.scatter(data_ex["measurement_date"],data_ex[column], color='b')
    ax.set_xlabel('time')
    ax.set_title(column)
    ax.set_xlim(time.min(),time.max())
    ax.legend()


# %% [markdown]
# 二度目のモデルの定義(学習データは正常データのみ)

# %%
#結果（日付と異常値スコアを含む配列の準備）
# 空のデータフレームを作成
results_df = pd.DataFrame(columns=["measurement_date", "anomaly_score"])
abnormal_df = pd.DataFrame(columns=["measurement_date", "anomaly_score"])

#初期値の変更回数
num_initializations = 5

#初期データの抜き出し及び、繰り返し処理の準備
data_trainestart = normal_df["measurement_date"][0]
data_trainestart_year = data_trainestart.year
data_trainestart_month = data_trainestart.month
data_trainestart = str(data_trainestart_year) + "/" + str(data_trainestart_month) + "/01 00:00:00"

data_trainend_year = data_trainestart_year + 2
data_trainend_month = data_trainestart_month
data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

data_teststart = data_trainend
if data_trainend_month == 12:
    data_testend_year = data_trainend_year + 1
    data_testend_month = 1
else:
    data_testend_year = data_trainend_year
    data_testend_month = data_trainend_month + 1

data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"


#モデル実装部分
for i in range(num):
    print("============================================")
    print(str(i+1) + "回目のモデル")
    #i回目のデータの抜き出し

    #トレーニングデータの抜き出し
    train_data = normal_df[(normal_df["measurement_date"]<data_trainend) & (normal_df["measurement_date"]>=data_trainestart)]
    train_data = train_data[colums_list]
    train_data = train_data.values


    #テストデータの抜き出し
    #テストデータはdata_trainendの一か月である
    test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
    test_data_origin = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
    test_data = test_data[colums_list]
    test_data = test_data.values

    #該当データ期間
    print("はじまり(train)" + str(data_trainestart))
    print("終わり(train)" + str(data_trainend))
    print("はじまり(test)" + str(data_teststart))
    print("終わり(test)" + str(data_testend))

    # 初期化ごとの最良のモデルを保持する変数
    best_model = None
    best_loss = np.inf

    #初期点変更による複数回モデルの作成
    for k in range(num_initializations):
        print(f"Initialization {k+1}/{num_initializations}")

        # モデルの作成
        model = model_autoencoder()

        #モデルの学習
        history = model.fit(train_data, train_data,
                epochs=10,
                shuffle=True,
                validation_data=(train_data, train_data))
        
        # 最良の損失値を保持し、最良のモデルを選ぶ
        val_loss = history.history['val_loss'][-1]
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
    


    #モデルの学習
    model.fit(train_data, train_data,
                epochs=5,
                shuffle=True,
                validation_data=(train_data, train_data))
    
    #結果
    result_data = model.predict(test_data)

    #pandas形式へ変換
    test_data = pd.DataFrame(test_data,columns=colums_list)
    print(len(test_data))
    result_data = pd.DataFrame(result_data,columns=colums_list)

    #異常値の算出
    abnormal_value = abnomalScores(test_data,result_data)



    #結果の格納
    temp_df = pd.DataFrame({
        "measurement_date": test_data_origin["measurement_date"].values,
        "anomaly_score": abnormal_value
    })

    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    #異常値のみの格納
    for j in range(len(abnormal_value)):
        if abnormal_value[j] > threshold:
            # 1行のデータフレームを作成
            tempab_df = pd.DataFrame({
                "measurement_date": [test_data_origin["measurement_date"].iloc[j]],
                "anomaly_score": [abnormal_value[j]]
            })
    
            # 異常値のみを格納するデータフレームに追加
            abnormal_df = pd.concat([abnormal_df, tempab_df], ignore_index=True)



    


    #日付の更新
    if data_trainestart_month == 12:
        data_trainestart_year += 1
        data_trainestart_month = 1
    else:
        data_trainestart_month += 1
    data_trainestart = str(data_trainestart_year) + "/" + str(data_trainestart_month) + "/01 00:00:00"        
    
    data_trainend_year = data_trainestart_year + 2
    data_trainend_month = data_trainestart_month
    data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"

    data_teststart = data_trainend
    if data_trainend_month == 12:
        data_testend_year = data_trainend_year + 1
        data_testend_month = 1
    else:
        data_testend_year = data_trainend_year
        data_testend_month = data_trainend_month + 1

    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

print(len(results_df))
print(len(abnormal_df))


    

# %% [markdown]
# 散布図の作成

# %%

# 散布図の作成
plt.figure(figsize=(80, 5))
plt.scatter(results_df["measurement_date"], results_df["anomaly_score"], c='blue', marker='o', edgecolor='k')
#異常スコアを赤で表示する
plt.scatter(abnormal_df["measurement_date"], abnormal_df["anomaly_score"], c='red', marker='o', edgecolor='k')

# y=0.8の位置に赤い横線を引く
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2)

# グラフのタイトルとラベル
plt.title('Anomaly Scores Over Time')
plt.xlabel('Measurement Date')
plt.ylabel('Anomaly Score')

# x軸の目盛りを自動調整して見やすくする
plt.xticks(rotation=45, ha='right')

# グラフを表示
plt.show()

# %%
#散布図の作成

#異常スコアの出力
#図の大きさを指定
fig = plt.figure(figsize=(60,100))

#はじめのサブプロットの作成（異常スコア）
ax1 = fig.add_subplot(13,1,1)
ax1.set_title("autoencoder_score")
#正常スコアと異常スコア
ax1.scatter(results_df["measurement_date"], results_df["anomaly_score"], c='blue', marker='o', edgecolor='k')
ax1.scatter(abnormal_df["measurement_date"], abnormal_df["anomaly_score"], c='red', marker='o', edgecolor='k')
ax1.set_ylabel('Abnormality')
# ラベルの設定
ax1.set_ylabel('Abnormality')
# y=0.8の位置に赤い横線を引く
ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2)

# 元の特徴量の時系列ごとのデータ
for i, column in enumerate (colums_list):
    ax = fig.add_subplot(13,1,i+2)
    ax.scatter(normal_df["measurement_date"],normal_df[column], color='b')
    ax.set_xlabel('time')
    ax.set_title(column)
    ax.set_xlim(time.min(),time.max())
    ax.legend()


# %%



