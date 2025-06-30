import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np




#散布図の作成（特徴量まで含めてすべて）
def plot_all(abnormal_scores,thresholds, colums_list, data_ex):

    #異常スコアの出力
    #図の大きさを指定
    fig = plt.figure(figsize=(60,150))
    #はじめのサブプロットの作成（異常スコア）
    ax1 = fig.add_subplot(13,1,1)

    ax1.set_title("Anomaly Score", fontsize=40)  
    ax1.set_xlabel("Time", fontsize=40)        
    ax1.set_ylabel("Anomaly Score", fontsize=40) 
    ax1.set_ylim(0, 1)

    # 軸の目盛りラベルサイズを変更する
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.tick_params(axis='x', labelsize=30)

    # 特定の日時を赤で表示する条件
    target_date = datetime.strptime("2018-08-25 04:00:03", "%Y-%m-%d %H:%M:%S")

    # 日付の範囲を取得
    min_date = abnormal_scores["measurement_date"].min()
    max_date = abnormal_scores["measurement_date"].max()

    # 色を格納するリスト
    colors = []

    # 各データポイントについて色を決定
    for _, row in abnormal_scores.iterrows():
        if row["measurement_date"] == target_date:
            colors.append("red")
        else:
            color_assigned = False
            for _, threshold in thresholds.iterrows():
                if threshold["test_start"] <= row["measurement_date"] <= threshold["test_end"]:
                    if row["anomaly_score"] > threshold["threshold_data"]:
                        colors.append("blue")
                    else:
                        colors.append("green")
                    color_assigned = True
                    break
            if not color_assigned:
                colors.append("gray")  # 該当する期間がない場合は灰色

    # 異常スコアの散布図
    sizes = [220 if color == "red" else 70 for color in colors]  # 赤い点はサイズを大きくする
    ax1.scatter(
        abnormal_scores["measurement_date"], 
        abnormal_scores["anomaly_score"], 
        c=colors, 
        s=sizes, 
        marker="o", 
        edgecolor="k", 
        label="Anomaly Score"
    )

    ax1.axhline(y=thresholds['threshold_data'].iloc[0],xmin=0, xmax=1,color='red', linestyle='--', linewidth = 1)






    #　各特徴量ごとの描写

    start_date = data_ex["measurement_date"].min()
    end_date = data_ex["measurement_date"].max()

    filtered_data = data_ex[
    (data_ex["measurement_date"] >= start_date) &
    (data_ex["measurement_date"] <= end_date)
    ]

    start_date = pd.to_datetime(filtered_data["measurement_date"].min())
    end_date = pd.to_datetime(filtered_data["measurement_date"].max())

    data_ex = data_ex[(data_ex["measurement_date"] >= start_date) & (data_ex["measurement_date"] <= end_date)]
    
    # 元の特徴量の時系列ごとのデータ
    for i, column in enumerate (colums_list):
        ax = fig.add_subplot(13,1,i+2)
        ax.scatter(data_ex["measurement_date"],data_ex[column], color='b')
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel('time', fontsize =30)
        ax.set_title(column, fontsize =30)
        ax.legend()



import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_abnormal_scores(abnormal_scores, thresholds, height):
    # 図の大きさを指定
    fig, ax = plt.subplots(figsize=(60, 15))
    ax.set_facecolor("#f5f5f5")  # 背景色を薄いグレーに設定
    fig.set_facecolor("#ffffff")  # プロットエリアを白に設定（任意）


    # 特定の日時を赤で表示する条件
    target_date = datetime.strptime("2018-08-25 04:00:03", "%Y-%m-%d %H:%M:%S")

    # 日付の範囲を取得
    min_date = abnormal_scores["measurement_date"].min()
    max_date = abnormal_scores["measurement_date"].max()

    # 色を格納するリスト
    colors = []

    # 各データポイントについて色を決定
    for _, row in abnormal_scores.iterrows():
        if row["measurement_date"] == target_date:
            colors.append("red")
        else:
            color_assigned = False
            for _, threshold in thresholds.iterrows():
                if threshold["test_start"] <= row["measurement_date"] <= threshold["test_end"]:
                    if row["anomaly_score"] > threshold["threshold_data"]:
                        colors.append("blue")
                    else:
                        colors.append("green")
                    color_assigned = True
                    break
            if not color_assigned:
                colors.append("gray")  # 該当する期間がない場合は灰色

    # 異常スコアの散布図
    sizes = [220 if color == "red" else 70 for color in colors]  # 赤い点はサイズを大きくする
    ax.scatter(
        abnormal_scores["measurement_date"], 
        abnormal_scores["anomaly_score"], 
        c=colors, 
        s=sizes, 
        marker="o", 
        edgecolor="k", 
        label="Anomaly Score"
    )

    # 閾値の線をプロット
    for _, threshold in thresholds.iterrows():
        test_start = threshold['test_start']
        test_end = threshold['test_end']

        # x 軸における相対位置を計算
        relative_start = (test_start - min_date) / (max_date - min_date)
        relative_end = (test_end - min_date) / (max_date - min_date)

        ax.axhline(
            y=threshold['threshold_data'],
            xmin=relative_start,
            xmax=relative_end,
            color="red",
            linestyle="--",
            linewidth=5  # 線をさらに太く
        )

    # 軸ラベルとタイトル
    ax.set_title("Anomaly Score", fontsize=80)
    ax.set_xlabel("Time", fontsize=70)
    ax.set_ylabel("Anomaly Score", fontsize=70)
    ax.set_ylim(0, height)  # 縦軸の最大値を0.2に固定

    # メモリフォントサイズを大きく
    ax.tick_params(axis='y', labelsize=60)
    ax.tick_params(axis='x', labelsize=60)

    # 凡例を表示
    ax.legend()

    # プロットを表示
    fig.tight_layout()
    plt.show()


def plot_train_scores(train_errors, y_scale):
    # 図の大きさを指定
    plt.figure(figsize=(60, 10))



    # 日付の範囲を取得
    min_date = train_errors["measurement_date"].min()
    max_date = train_errors["measurement_date"].max()

    # 色を格納するリスト
    colors = []

    colors.append("blue")

    # 異常スコアの散布図
    sizes = [220 if color == "red" else 70 for color in colors]  # 赤い点はサイズを大きくする
    plt.scatter(
        train_errors["measurement_date"], 
        train_errors["errors_score"], 
        c=colors, 
        s=sizes, 
        marker="o", 
        edgecolor="k", 
        label="Anomaly Score"
    )

    # 軸ラベルとタイトル
    plt.title("Anomaly Score", fontsize=40)
    plt.xlabel("Time", fontsize=40)
    plt.ylabel("Anomaly Score", fontsize=40)
    plt.ylim(0, y_scale)  # 縦軸の最大値を0.2に固定

    # 縦軸のメモリフォントサイズを大きく
    plt.yticks(fontsize=25)

    # 横軸の日付ラベルのフォントサイズを調整
    plt.xticks(fontsize=40, rotation=45)  # フォントサイズを15にし、ラベルを45度回転

    # 凡例を表示
    plt.legend()

    # プロットを表示
    plt.tight_layout()
    plt.show()



def plot_errors_per_features(errors_per_features):
    plt.figure(figsize=(12,6))
    labels = errors_per_features.columns
    data = errors_per_features.iloc[0]

    width = 0.8
    plt.xticks(rotation = 90)
    plt.xlabel("features")
    plt.ylabel("errors")
    plt.ylim(0, 0.005)
    plt.bar(labels, data, width=width)
    plt.show()

def plot_errors_per_data(errors_per_datas, xlim,ylim):

    fig, ax = plt.subplots()

    # ビン幅を0.1に設定
    bins = np.arange(start=min(errors_per_datas.loc[:,"errors_score"]), stop=max(errors_per_datas.loc[:,"errors_score"])+0.0002, step=0.0002)

    ax.hist(errors_per_datas.loc[:,"errors_score"],bins=bins)
    ax.set_title("errors_per_data")
    ax.set_xlabel("errors")
    ax.set_ylabel("freq")
    ax.set_ylim(0, ylim)
    ax.set_xlim(0,xlim)
    fig.show()


