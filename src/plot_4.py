import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np




#散布図の作成（特徴量まで含めてすべて）
def plot_all(abnormal_scores,thresholds, colums_list, data_ex):

    #異常スコアの出力
    #図の大きさを指定
    fig = plt.figure(figsize=(60,100))
    #はじめのサブプロットの作成（異常スコア）
    ax1 = fig.add_subplot(13,1,1)
    ax1.set_title("autoencoder_score")
    ax1.set_xlabel('time',fontsize=25)
    # 軸の目盛りラベルサイズを変更する
    ax1.tick_params(axis='both', which='major', labelsize=30)

    #正常スコアと異常スコア
    ax1.scatter(abnormal_scores["measurement_date"], abnormal_scores["anomaly_score"], c='blue', marker='o', edgecolor='k')
    # 各期間の異常スコアの閾値を描画

    # measurement_date の最小値と最大値を取得

    min_date = abnormal_scores["measurement_date"].min()
    max_date = abnormal_scores["measurement_date"].max()
    for term, threshold in thresholds.items():
        # test_start と test_end を datetime 型に変換
        test_start = pd.to_datetime(threshold['train_start'])
        test_end = pd.to_datetime(threshold['train_end'])
        relative_start = (test_start - min_date) / (max_date - min_date)
        relative_end = (test_end - min_date) / (max_date - min_date)
        
        ax1.axhline(y=threshold['threshold_data'],xmin=relative_start, xmax=relative_end,color='red', linestyle='--', label=f"Threshold (Term {term})", linewidth = 1)
    
    # ラベルの設定
    ax1.set_ylabel('Abnormality')


    start_date = pd.to_datetime("2018-06-01")
    end_date = pd.to_datetime("2018-09-01")
    data_ex = data_ex[(data_ex["measurement_date"] >= start_date) & (data_ex["measurement_date"] <= end_date)]
    
    # 元の特徴量の時系列ごとのデータ
    for i, column in enumerate (colums_list):
        ax = fig.add_subplot(13,1,i+2)
        ax.scatter(data_ex["measurement_date"],data_ex[column], color='b')
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel('time')
        ax.set_title(column)
        ax.legend()



import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def plot_abnormal_scores(abnormal_scores, thresholds):
    # 図の大きさを指定
    plt.figure(figsize=(60, 10))

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
    plt.scatter(
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

        plt.axhline(
            y=threshold['threshold_data'],
            xmin=relative_start,
            xmax=relative_end,
            color="red",
            linestyle="--",
            linewidth=5  # 線をさらに太く
        )

    # 軸ラベルとタイトル
    plt.title("Anomaly Score", fontsize=40)
    plt.xlabel("Time", fontsize=40)
    plt.ylabel("Anomaly Score", fontsize=40)
    plt.ylim(0, 0.20)  # 縦軸の最大値を0.2に固定

    # 縦軸のメモリフォントサイズを大きく
    plt.yticks(fontsize=25)

    # 横軸の日付ラベルのフォントサイズを調整
    plt.xticks(fontsize=40, rotation=45)  # フォントサイズを15にし、ラベルを45度回転

    # 凡例を表示
    plt.legend()

    # プロットを表示
    plt.tight_layout()
    plt.show()
