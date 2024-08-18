
#必要ライブラリのインストール
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

# 生データの読み込み
# ECDISデータの読み込み
raw_data_ECDIS = pd.read_csv('../data/ECDIS__FMD-3x00.csv', header=0, delimiter=',',dtype = {"equipment_label_no":"object","f_shipno":"object"})

print("データの読み込み完了")

raw_data_ECDIS = raw_data_ECDIS >> mutate(id = X.f_shipno+"-"+X.equipment_label_no,
                         measurement_ymd = X.measurement_date.str[:10],
                         measurement_ymd_h = X.measurement_date.str[11:13]) >> mutate(
    measurement_ymd_hms = X.measurement_ymd+" "+X.measurement_ymd_h+":00:00")
