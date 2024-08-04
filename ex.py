import pandas as pd

# CSVファイルのパス
ecdis = 'data\ECDIS__FMD-3x00.csv'
radar = "data\Radar__FAR-3xx0.csv"

# CSVファイルを読み込む
df1 = pd.read_csv(ecdis)
df2 = pd.read_csv(radar)

# 最初の10行を表示する
print(df1.head(10))
print(df2.head(10))
