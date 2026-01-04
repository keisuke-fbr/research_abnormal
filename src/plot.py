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



# plot.py
# 異常検知実験結果の可視化モジュール

# plot.py
# 異常検知実験結果の可視化モジュール

# plot.py
# 異常検知実験結果の可視化モジュール

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# フォント設定
# =============================================================================

def setup_japanese_font():
    """
    日本語フォントの設定
    Windows環境ではMSゴシック、その他ではIPAexゴシックを使用
    """
    # 日本語フォントの候補リスト
    font_candidates = [
        'MS Gothic',           # Windows
        'MS ゴシック',          # Windows (日本語名)
        'Yu Gothic',           # Windows
        'Hiragino Sans',       # macOS
        'IPAexGothic',         # Linux
        'IPAGothic',           # Linux
        'Noto Sans CJK JP',    # Linux
        'DejaVu Sans',         # フォールバック
    ]
    
    # 使用可能なフォントを探す
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            break
    
    # マイナス記号の文字化け対策
    plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 可視化関数
# =============================================================================

def plot_anomaly_scores(abnormal_scores, thresholds, y_max=0.2):
    """
    異常データの可視化
    
    入力:
        abnormal_scores: DataFrame - 異常スコアデータ（measurement_date, anomaly_score）
        thresholds: DataFrame - 閾値データ（test_start, test_end, threshold_data）
        y_max: float - 縦軸の最大値
    
    備考:
        - 2018-08-25 04:00:03のデータは赤点でプロット
        - その他のデータは青点でプロット
        - 閾値は赤の点線で期間ごとに表示
        - 閾値がない期間は前の期間の閾値を適用
    """
    setup_japanese_font()
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 特定の日時（赤点でプロット）
    target_date = datetime.strptime("2018-08-25 04:00:03", "%Y-%m-%d %H:%M:%S")
    
    # 日付の範囲を取得
    min_date = abnormal_scores["measurement_date"].min()
    max_date = abnormal_scores["measurement_date"].max()
    
    # 赤点と青点に分類
    red_mask = abnormal_scores["measurement_date"] == target_date
    blue_mask = ~red_mask
    
    # 青点のプロット
    ax.scatter(
        abnormal_scores.loc[blue_mask, "measurement_date"],
        abnormal_scores.loc[blue_mask, "anomaly_score"],
        c="blue",
        s=30,
        marker="o",
        label="異常スコア"
    )
    
    # 赤点のプロット（特定日時）
    if red_mask.any():
        ax.scatter(
            abnormal_scores.loc[red_mask, "measurement_date"],
            abnormal_scores.loc[red_mask, "anomaly_score"],
            c="red",
            s=100,
            marker="o",
            edgecolor="black",
            linewidth=1,
            label="特定日時 (2018-08-25 04:00:03)",
            zorder=5
        )
    
    # 閾値データを日付順にソート
    thresholds_sorted = thresholds.sort_values("test_start").reset_index(drop=True)
    
    # 全期間をカバーするために拡張された閾値リストを作成
    extended_thresholds = []
    
    # 閾値データの最初の期間より前の期間がある場合
    if len(thresholds_sorted) > 0:
        first_threshold_start = thresholds_sorted.iloc[0]["test_start"]
        if min_date < first_threshold_start:
            # 最初の閾値を適用
            extended_thresholds.append({
                "test_start": min_date,
                "test_end": first_threshold_start,
                "threshold_data": thresholds_sorted.iloc[0]["threshold_data"]
            })
    
    # 既存の閾値を追加し、閾値間のギャップを埋める
    prev_threshold_value = None
    prev_threshold_end = None
    
    for idx, threshold in thresholds_sorted.iterrows():
        test_start = threshold["test_start"]
        test_end = threshold["test_end"]
        threshold_value = threshold["threshold_data"]
        
        # 前の閾値の終了と現在の閾値の開始の間にギャップがある場合
        if prev_threshold_end is not None and test_start > prev_threshold_end:
            extended_thresholds.append({
                "test_start": prev_threshold_end,
                "test_end": test_start,
                "threshold_data": prev_threshold_value
            })
        
        # 現在の閾値を追加
        extended_thresholds.append({
            "test_start": test_start,
            "test_end": test_end,
            "threshold_data": threshold_value
        })
        
        prev_threshold_value = threshold_value
        prev_threshold_end = test_end
    
    # 閾値データの最後の期間より後の期間がある場合
    if prev_threshold_end is not None and max_date > prev_threshold_end:
        extended_thresholds.append({
            "test_start": prev_threshold_end,
            "test_end": max_date,
            "threshold_data": prev_threshold_value
        })
    
    # 閾値の線をプロット
    total_range = (max_date - min_date).total_seconds()
    
    for threshold in extended_thresholds:
        test_start = threshold["test_start"]
        test_end = threshold["test_end"]
        
        if total_range > 0:
            relative_start = (test_start - min_date).total_seconds() / total_range
            relative_end = (test_end - min_date).total_seconds() / total_range
            
            # 範囲を0-1に制限
            relative_start = max(0, min(1, relative_start))
            relative_end = max(0, min(1, relative_end))
            
            ax.axhline(
                y=threshold["threshold_data"],
                xmin=relative_start,
                xmax=relative_end,
                color="red",
                linestyle="--",
                linewidth=2
            )
    
    # 凡例用に閾値の線を1本だけ追加
    ax.axhline(y=-1, color="red", linestyle="--", linewidth=2, label="閾値")
    
    # 軸ラベルとタイトル
    ax.set_title("異常検知結果", fontsize=20)
    ax.set_xlabel("時刻", fontsize=16)
    ax.set_ylabel("再構成誤差", fontsize=16)
    ax.set_ylim(0, y_max)
    
    # 目盛りのフォントサイズ
    ax.tick_params(axis='both', labelsize=12)
    
    # 凡例（重複を除去）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    
    # グリッド
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.show()


def plot_feature_avg_errors(errors_avg, y_max=0.01):
    """
    特徴平均誤差の可視化（縦棒グラフ）
    
    入力:
        errors_avg: DataFrame - 特徴量ごとの平均再構成誤差
        y_max: float - 縦軸の最大値
    """
    setup_japanese_font()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = errors_avg.columns
    data = errors_avg.iloc[0]
    
    x = np.arange(len(labels))
    width = 0.7
    
    bars = ax.bar(x, data, width=width, color='steelblue', edgecolor='black')
    
    ax.set_title("学習完了時の各特徴ごとの再構成誤差（平均）", fontsize=18)
    ax.set_xlabel("特徴名", fontsize=14)
    ax.set_ylabel("再構成誤差", fontsize=14)
    ax.set_ylim(0, y_max)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.show()


def plot_feature_max_errors(errors_max, y_max=0.01):
    """
    特徴最大誤差の可視化（縦棒グラフ）
    
    入力:
        errors_max: DataFrame - 閾値データの各特徴ごとの再構成誤差
        y_max: float - 縦軸の最大値
    """
    setup_japanese_font()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = errors_max.columns
    data = errors_max.iloc[0]
    
    x = np.arange(len(labels))
    width = 0.7
    
    bars = ax.bar(x, data, width=width, color='coral', edgecolor='black')
    
    ax.set_title("再構成誤差最大データの各特徴ごとの再構成誤差", fontsize=18)
    ax.set_xlabel("特徴名", fontsize=14)
    ax.set_ylabel("再構成誤差", fontsize=14)
    ax.set_ylim(0, y_max)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.show()


def plot_error_histogram(errors_per_data, bin_width=0.001, y_max=100):
    """
    学習データの誤差分布可視化（ヒストグラム）
    
    入力:
        errors_per_data: DataFrame - データごとの再構成誤差（errors_scoreカラム）
        bin_width: float - ビン幅
        y_max: float - 縦軸の最大値
    """
    setup_japanese_font()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    errors = errors_per_data["errors_score"]
    
    # ビンの作成
    bins = np.arange(
        start=errors.min(),
        stop=errors.max() + bin_width,
        step=bin_width
    )
    
    ax.hist(errors, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_title("学習データの再構成誤差分布", fontsize=18)
    ax.set_xlabel("再構成誤差", fontsize=14)
    ax.set_ylabel("頻度", fontsize=14)
    ax.set_ylim(0, y_max)
    
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.show()


def _plot_colored_line(ax, x, y, threshold, below_color='blue', above_color='red', linewidth=1.5):
    """
    閾値を境に色が変わる折れ線グラフを描画
    
    入力:
        ax: Axes - 描画先のAxes
        x: array - x座標
        y: array - y座標
        threshold: float - 閾値
        below_color: str - 閾値以下の色
        above_color: str - 閾値以上の色
        linewidth: float - 線の太さ
    """
    x = np.array(x)
    y = np.array(y)
    
    # セグメントの作成
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 各セグメントの色を決定（両端の平均値で判定）
    segment_means = (y[:-1] + y[1:]) / 2
    colors = [above_color if mean >= threshold else below_color for mean in segment_means]
    
    # LineCollectionで描画
    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    
    # 軸の範囲を自動調整
    ax.autoscale()


def plot_learning_progress(learning_scores, columns_list, feature_thresholds, y_max_avg=0.5, y_max_max=0.5):
    """
    学習推移の可視化（n行2列：左が平均誤差、右が最大誤差）
    
    入力:
        learning_scores: DataFrame - 学習ログデータ（epoch, 特徴名_avg, 特徴名_max）
        columns_list: list - 特徴量名リスト
        feature_thresholds: dict - 特徴量名→基準値のマッピング（例: {"feature1": 0.1, "feature2": 0.1}）
        y_max_avg: float - 平均誤差グラフの縦軸の最大値
        y_max_max: float - 最大誤差グラフの縦軸の最大値
    
    備考:
        - 最大誤差グラフには基準値を緑点線で表示
        - 最大誤差の折れ線は基準値以下が青、基準値以上が赤
    """
    setup_japanese_font()
    
    num_features = len(columns_list)
    
    fig, axes = plt.subplots(num_features, 2, figsize=(20, 4 * num_features))
    
    epochs = learning_scores["epoch"].values
    
    for row_idx, col_name in enumerate(columns_list):
        # 基準値を取得
        threshold = feature_thresholds.get(col_name, 0.1)
        
        # 左列：avg のプロット（通常の青線）
        ax_avg = axes[row_idx, 0]
        avg_col = f"{col_name}_avg"
        if avg_col in learning_scores.columns:
            ax_avg.plot(epochs, learning_scores[avg_col], color='blue', linewidth=1.5)
        ax_avg.set_title(f"{col_name} - 平均誤差", fontsize=12)
        ax_avg.set_xlabel("エポック数", fontsize=10)
        ax_avg.set_ylabel("再構成誤差", fontsize=10)
        ax_avg.set_ylim(0, y_max_avg)
        ax_avg.grid(True, alpha=0.3)
        ax_avg.tick_params(axis='both', labelsize=9)
        
        # 右列：max のプロット（基準値で色分け）
        ax_max = axes[row_idx, 1]
        max_col = f"{col_name}_max"
        if max_col in learning_scores.columns:
            y_values = learning_scores[max_col].values
            
            # 基準値を緑点線で表示
            ax_max.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'基準値 ({threshold})')
            
            # 色分け折れ線グラフを描画
            _plot_colored_line(ax_max, epochs, y_values, threshold, 
                              below_color='blue', above_color='red', linewidth=1.5)
        
        ax_max.set_title(f"{col_name} - 最大誤差", fontsize=12)
        ax_max.set_xlabel("エポック数", fontsize=10)
        ax_max.set_ylabel("再構成誤差", fontsize=10)
        ax_max.set_ylim(0, y_max_max)
        ax_max.set_xlim(epochs.min(), epochs.max())
        ax_max.grid(True, alpha=0.3)
        ax_max.tick_params(axis='both', labelsize=9)
        
        # 凡例（最初の行のみ）
        if row_idx == 0:
            ax_max.legend(fontsize=9)
    
    fig.tight_layout()
    plt.show()