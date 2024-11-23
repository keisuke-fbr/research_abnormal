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
        test_start = pd.to_datetime(threshold['test_start'])
        test_end = pd.to_datetime(threshold['test_end'])
        relative_start = (test_start - min_date) / (max_date - min_date)
        relative_end = (test_end - min_date) / (max_date - min_date)
        
        ax1.axhline(y=threshold['threshold_data'],xmin=relative_start, xmax=relative_end,color='red', linestyle='--', label=f"Threshold (Term {term})", linewidth = 10)
    
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







def data_describe(traindata_model_df, data_ex):

    print("データの構造確認")

    print("元データ")
    print(data_ex.describe())

    print("========================================================================================")
    print("========================================================================================")

    print("再構築データ")
    print(traindata_model_df.describe())


import matplotlib.pyplot as plt

# ユニット数の合計を横軸に、final_lossを縦軸に描写する関数
def plot_final_loss_vs_unit_sum(final_losses_per_units):
    unit_sums = []
    losses = []
    
    # 各 (units_1_3, units_2) に対して合計ユニット数とfinal_lossを取得
    for (units_1_3, units_2), avg_loss in final_losses_per_units.items():
        unit_sum = units_1_3 + units_2  # ユニット数の合計
        unit_sums.append(unit_sum)
        losses.append(avg_loss)
    
    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(unit_sums, losses)
    plt.title('Final Loss vs Total Unit Count')
    plt.xlabel('Total Unit Count (units_1_3 + units_2)')
    plt.ylabel('Final Loss')
    plt.grid(True)
    plt.show()


def plot_averages(loss_array):
    """
    `loss_array` は以下の形式の三次元配列:
    {
        units2: {
            units1_3: 平均損失 (float) or None
        }
    }
    """
    # 手動で色を定義（9色）
    colors = [
        "blue", "orange", "green", "red", "purple", 
        "brown", "pink", "gray", "olive"
    ]
    
    plt.figure(figsize=(10, 6))
    
    # 各 units2 ごとにプロット
    for index, (units2, units1_3_dict) in enumerate(loss_array.items()):
        x = []
        y = []
        
        # units1_3 と対応する損失値をリストに取得（None は除外）
        for units1_3, loss in units1_3_dict.items():
            if loss is not None:
                x.append(units1_3)  # 横軸に units1_3 を設定
                y.append(loss)      # 縦軸に平均損失を設定
        
        # データが存在しない場合はスキップ
        if not x or not y:
            continue
        
        # プロット（`units2` ごとに手動で設定した色を使用）
        color = colors[index % len(colors)]  # 9色をループする
        plt.plot(x, y, marker='o', label=f"units2 = {units2}", color=color)
    
    # プロットの装飾
    plt.title("Combined Loss Plot for Different units2")
    plt.xlabel("units1_3")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


#　トレーニングデータにおける特徴量ごとの再構成誤差をプロット
def plot_error_train(errors_train):
    # 各手法のプロット

    # プロットの設定
    methods = ['decrease', 'skip_decrease', 'binary']  # 各手法
    periods = list(errors_train.keys())  # 各 period
    colors = ['b', 'g', 'r', 'c']  # 各 period の色
    feature_names = errors_train[0]['decrease'].index  # 特徴量名

    for method in methods:
        plt.figure(figsize=(12,6))
        plt.title(f"Reconstruction Error by Feature - {method}")
        plt.xlabel("Features")
        plt.ylabel("Reconstruction Error")

        feature_data = {}

        for i, feature_name in enumerate(feature_names):
            feature_data[feature_name] = []

        for i, period in enumerate(periods):
            for k, feature_name in enumerate(feature_names):
                feature_data[feature_name].append(errors_train[period][method][feature_name])

        feature_data_ave = {feature:np.mean(values) for feature, values in feature_data.items()}

        # 各 period を色分けしてプロット
        for i, period in enumerate(periods):
            # 特徴量の再構成誤差を取得
            errors = errors_train[period][method]
        
            # プロット
            plt.plot(feature_names, errors, marker='o', linestyle='-', color=colors[i], label=f'Period {period}')
    
        plt.plot(feature_names, list(feature_data_ave.values()), marker='o', linestyle = '-', color='black', label = 'average', lw = "3.0")

        plt.xticks(rotation=45)  # 特徴量名を見やすくするために回転
        plt.legend()  # 凡例の表示
        plt.tight_layout()  # レイアウト調整
        plt.show()


#　テストデータにおける特徴量ごとの再構成誤差をプロット
def plot_error_test(errors_predict):
# 各手法のプロット

    # プロットの設定
    methods = ['decrease', 'skip_decrease', 'binary']  # 各手法
    periods = list(errors_predict.keys())  # 各 period
    colors = ['b', 'g', 'r', 'c']  # 各 period の色
    feature_names = errors_predict[0]['decrease'].index  # 特徴量名

    for method in methods:
        plt.figure(figsize=(12,6))
        plt.title(f"Reconstruction Error by Feature - {method}")
        plt.xlabel("Features")
        plt.ylabel("Reconstruction Error")

        feature_data = {}

        for i, feature_name in enumerate(feature_names):
            feature_data[feature_name] = []

        for i, period in enumerate(periods):
            for k, feature_name in enumerate(feature_names):
                feature_data[feature_name].append(errors_predict[period][method][feature_name])

        feature_data_ave = {feature:np.mean(values) for feature, values in feature_data.items()}

        # 各 period を色分けしてプロット
        for i, period in enumerate(periods):
            # 特徴量の再構成誤差を取得
            errors = errors_predict[period][method]
        
            # プロット
            plt.plot(feature_names, errors, marker='o', linestyle='-', color=colors[i], label=f'Period {period}')
    
        plt.plot(feature_names, list(feature_data_ave.values()), marker='o', linestyle = '-', color='black', label = 'average', lw = "3.0")

        plt.xticks(rotation=45)  # 特徴量名を見やすくするために回転
        plt.legend()  # 凡例の表示
        plt.tight_layout()  # レイアウト調整
        plt.show()

#閾値のみのプロット
def plot_threshold(abnormal_values):
    # 各手法のプロット

    # プロットの設定
    methods = ['decrease', 'skip_decrease', 'binary']  # 各手法
    periods = list(abnormal_values.keys())  # 各 period

    plt.figure(figsize=(12,6))
    plt.title(f"閾値")
    plt.xlabel("period")
    plt.ylabel("閾値")


    for method in methods:
        values = {}
        for period in periods:
            values[period] = abnormal_values[period][method]

        plt.plot(periods, list(values.values()), marker='o', linestyle='-', label=f'the way:{method}')
    plt.xticks(rotation=45)  # 特徴量名を見やすくするために回転
    plt.legend()  # 凡例の表示
    plt.tight_layout()  # レイアウト調整
    plt.show()
