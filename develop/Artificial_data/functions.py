"""
センサ人工データ生成用の関数群
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windows用
# plt.rcParams['font.family'] = 'Hiragino Sans'  # Mac用
# plt.rcParams['font.family'] = 'IPAexGothic'  # Linux用
plt.rcParams['axes.unicode_minus'] = False


def generate_sensor_data(
    start_date: datetime,
    end_date: datetime,
    interval_hours: int,
    mean: float,
    period_days: float,
    amplitude: float,
    std: float,
    seed: int = None
) -> np.ndarray:
    """1センサ分の人工データを生成する"""
    if seed is not None:
        np.random.seed(seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{interval_hours}h')
    n_points = len(date_range)
    t_days = np.arange(n_points) * interval_hours / 24
    
    signal = mean + amplitude * np.sin(2 * np.pi * t_days / period_days)
    noise = np.random.normal(0, std, n_points)
    
    return signal + noise


def generate_single_point(
    index: int,
    interval_hours: int,
    mean: float,
    period_days: float,
    amplitude: float,
    std: float,
    seed: int = None
) -> float:
    """指定インデックスの1点を生成する"""
    if seed is not None:
        np.random.seed(seed)
    
    t_days = index * interval_hours / 24
    signal = mean + amplitude * np.sin(2 * np.pi * t_days / period_days)
    noise = np.random.normal(0, std)
    
    return signal + noise


def generate_dataset(
    start_date: datetime,
    end_date: datetime,
    interval_hours: int,
    sensor_params: list[dict],
    seed: int = None
) -> pd.DataFrame:
    """複数センサのデータセットを生成する（センサごとのflag列付き）"""
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{interval_hours}h')
    
    df = pd.DataFrame({'measurement_date': date_range})
    
    for i, params in enumerate(sensor_params, start=1):
        sensor_seed = seed + i if seed is not None else None
        values = generate_sensor_data(
            start_date=start_date,
            end_date=end_date,
            interval_hours=interval_hours,
            mean=params['mean'],
            period_days=params['period_days'],
            amplitude=params['amplitude'],
            std=params['std'],
            seed=sensor_seed
        )
        df[f'sensor{i}'] = values
        # 各センサごとにflag列を追加（初期値は0=正常）
        df[f'sensor{i}_flag'] = 0
    
    return df


def apply_anomalies(
    df: pd.DataFrame,
    interval_hours: int,
    sensor_params: list[dict],
    anomaly_settings: dict,
    precursor_settings: dict,
    seed: int = None
) -> pd.DataFrame:
    """
    異常・予兆データを適用する
    
    Parameters
    ----------
    df : pd.DataFrame
        正常データのDataFrame
    interval_hours : int
        取得間隔（時間）
    sensor_params : list[dict]
        各センサの元パラメータ
    anomaly_settings : dict
        {sensor_idx: [(datetime, mean), ...], ...}
    precursor_settings : dict
        {sensor_idx: [(datetime, mean), ...], ...}
    seed : int, optional
        乱数シード
    
    Returns
    -------
    pd.DataFrame
        異常・予兆が適用されたDataFrame
    """
    df = df.copy()
    
    # 異常データ適用 (flag=2)
    for sensor_idx, settings in anomaly_settings.items():
        sensor_col = f'sensor{sensor_idx + 1}'
        flag_col = f'sensor{sensor_idx + 1}_flag'
        params = sensor_params[sensor_idx]
        
        for dt, new_mean in settings:
            mask = df['measurement_date'] == pd.Timestamp(dt)
            if mask.any():
                idx = df[mask].index[0]
                point_seed = seed + sensor_idx * 1000 + idx if seed is not None else None
                new_value = generate_single_point(
                    index=idx,
                    interval_hours=interval_hours,
                    mean=new_mean,
                    period_days=params['period_days'],
                    amplitude=params['amplitude'],
                    std=params['std'],
                    seed=point_seed
                )
                df.loc[mask, sensor_col] = new_value
                df.loc[mask, flag_col] = 2
    
    # 予兆データ適用 (flag=1、ただし既に異常(2)なら上書きしない)
    for sensor_idx, settings in precursor_settings.items():
        sensor_col = f'sensor{sensor_idx + 1}'
        flag_col = f'sensor{sensor_idx + 1}_flag'
        params = sensor_params[sensor_idx]
        
        for dt, new_mean in settings:
            mask = df['measurement_date'] == pd.Timestamp(dt)
            if mask.any():
                idx = df[mask].index[0]
                # 異常でない場合のみ予兆として設定
                if df.loc[mask, flag_col].values[0] != 2:
                    point_seed = seed + sensor_idx * 1000 + idx + 500 if seed is not None else None
                    new_value = generate_single_point(
                        index=idx,
                        interval_hours=interval_hours,
                        mean=new_mean,
                        period_days=params['period_days'],
                        amplitude=params['amplitude'],
                        std=params['std'],
                        seed=point_seed
                    )
                    df.loc[mask, sensor_col] = new_value
                    df.loc[mask, flag_col] = 1
    
    return df


def plot_sensor_data(
    df: pd.DataFrame,
    title: str = "センサデータ",
    figsize_per_sensor: tuple = (12, 3),
    y_min: float = None,
    y_max: float = None
) -> plt.Figure:
    """センサデータを可視化する（センサごとのflag列で色分け）
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    title : str
        グラフタイトル
    figsize_per_sensor : tuple
        1センサあたりの図のサイズ
    y_min : float, optional
        Y軸の下限値
    y_max : float, optional
        Y軸の上限値
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor') and not col.endswith('_flag')]
    n_sensors = len(sensor_cols)
    
    fig, axes = plt.subplots(
        nrows=n_sensors,
        ncols=1,
        figsize=(figsize_per_sensor[0], figsize_per_sensor[1] * n_sensors),
        sharex=True
    )
    
    if n_sensors == 1:
        axes = [axes]
    
    # 色設定
    colors = {0: 'blue', 1: 'orange', 2: 'red'}
    labels = {0: '正常', 1: '予兆', 2: '異常'}
    
    for ax, col in zip(axes, sensor_cols):
        flag_col = f'{col}_flag'
        
        # センサごとのflag列がある場合は色分け
        if flag_col in df.columns:
            for flag_val in [0, 1, 2]:
                mask = df[flag_col] == flag_val
                if mask.any():
                    if flag_val == 0:
                        ax.plot(
                            df.loc[mask, 'measurement_date'],
                            df.loc[mask, col],
                            'o',
                            markersize=1,
                            color=colors[flag_val],
                            label=labels[flag_val]
                        )
                    else:
                        ax.plot(
                            df.loc[mask, 'measurement_date'],
                            df.loc[mask, col],
                            'o',
                            markersize=5,
                            color=colors[flag_val],
                            label=labels[flag_val]
                        )
            ax.legend(loc='upper right')
        else:
            ax.plot(df['measurement_date'], df[col], 'o', markersize=1, color='blue')
        
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        
        # Y軸の範囲設定
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
    
    axes[0].set_title(title)
    axes[-1].set_xlabel('日付')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def save_dataset(
    df: pd.DataFrame,
    output_dir: str = "./data",
    filename: str = None
) -> str:
    """データセットをCSVファイルとして保存する（flag列を統合）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # センサ列とflag列を分離
    sensor_cols = [col for col in df.columns if col.startswith('sensor') and not col.endswith('_flag')]
    flag_cols = [col for col in df.columns if col.endswith('_flag')]
    
    # 保存用DataFrameを作成
    save_df = df[['measurement_date'] + sensor_cols].copy()
    
    # 各センサのflag列を統合（異常=2 > 予兆=1 > 正常=0）
    if flag_cols:
        save_df['flag'] = df[flag_cols].max(axis=1).astype(int)
    else:
        save_df['flag'] = 0
    
    if filename is None:
        n_sensors = len(sensor_cols)
        filename = f"artificial_data_sensor_{n_sensors}.csv"
    
    filepath = output_path / filename
    save_df.to_csv(filepath, index=False)
    
    return str(filepath)


# ========== UI関連 ==========

class SensorDataGeneratorUI:
    """センサデータ生成用のUIクラス"""
    
    def __init__(self):
        self.generated_df = None
        self.sensor_params = []
        self.date_options = []
        self.interval_hours = 1
        self._build_ui()
    
    def _build_ui(self):
        """UIコンポーネントを構築"""
        # 共通パラメータ
        self.start_date_picker = widgets.DatePicker(
            description='開始日:',
            value=datetime(2023, 1, 1).date()
        )
        self.end_date_picker = widgets.DatePicker(
            description='終了日:',
            value=datetime(2025, 3, 31).date()
        )
        self.interval_input = widgets.IntText(
            value=1,
            description='間隔(時間):',
            style={'description_width': 'initial'}
        )
        
        # Y軸範囲（複数センサ用・共通）
        self.y_min_input = widgets.FloatText(
            value=0.0,
            description='Y軸下限:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='150px')
        )
        self.y_max_input = widgets.FloatText(
            value=200.0,
            description='Y軸上限:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='150px')
        )
        
        # センサ数
        self.n_sensors_input = widgets.IntSlider(
            value=5, min=1, max=10, step=1,
            description='センサ数:'
        )
        
        # 動的パラメータ領域
        self.sensor_params_container = widgets.VBox()
        self.anomaly_settings_container = widgets.VBox()
        self.output_area = widgets.Output()
        
        # ボタン
        self.generate_normal_button = widgets.Button(description='正常データ生成', button_style='primary')
        self.apply_anomaly_button = widgets.Button(description='異常データ適用', button_style='warning')
        self.save_button = widgets.Button(description='保存', button_style='success')
        
        # 初期状態で非表示
        self.apply_anomaly_button.layout.display = 'none'
        self.anomaly_settings_container.layout.display = 'none'
        
        # イベント登録
        self.n_sensors_input.observe(self._update_sensor_params, names='value')
        self.generate_normal_button.on_click(self._on_generate_normal_click)
        self.apply_anomaly_button.on_click(self._on_apply_anomaly_click)
        self.save_button.on_click(self._on_save_click)
        
        # 初期化
        self._update_sensor_params(self.n_sensors_input.value)
    
    def _create_sensor_param_widget(self, sensor_id: int) -> widgets.VBox:
        """1センサ分のパラメータウィジェットを作成"""
        return widgets.VBox([
            widgets.HTML(f"<b>センサ{sensor_id}</b>"),
            widgets.HBox([
                widgets.FloatText(value=100.0, description='平均:', layout=widgets.Layout(width='150px')),
                widgets.FloatText(value=30, description='周期(日):', layout=widgets.Layout(width='150px')),
                widgets.FloatText(value=10.0, description='振幅:', layout=widgets.Layout(width='150px')),
                widgets.FloatText(value=10.0, description='標準偏差:', layout=widgets.Layout(width='150px')),
            ])
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px 0'))
    
    def _update_sensor_params(self, change):
        """センサ数変更時にパラメータ入力欄を更新"""
        n = change['new'] if isinstance(change, dict) else change
        self.sensor_params_container.children = [
            self._create_sensor_param_widget(i) for i in range(1, n + 1)
        ]
    
    def _get_sensor_params(self) -> list[dict]:
        """入力されたパラメータを取得"""
        params = []
        for sensor_box in self.sensor_params_container.children:
            hbox = sensor_box.children[1]
            params.append({
                'mean': hbox.children[0].value,
                'period_days': hbox.children[1].value,
                'amplitude': hbox.children[2].value,
                'std': hbox.children[3].value
            })
        return params
    
    def _create_anomaly_point_widget(self, point_id: int, sensor_id: int, anomaly_type: str) -> widgets.HBox:
        """1異常/予兆ポイントの入力ウィジェットを作成"""
        return widgets.HBox([
            widgets.HTML(f"{anomaly_type}{point_id}: "),
            widgets.Dropdown(
                options=self.date_options,
                description='日時:',
                layout=widgets.Layout(width='250px')
            ),
            widgets.FloatText(
                value=150.0,
                description='平均:',
                layout=widgets.Layout(width='150px')
            )
        ])
    
    def _create_sensor_anomaly_widget(self, sensor_id: int) -> widgets.VBox:
        """1センサ分の異常・予兆設定ウィジェットを作成"""
        # 異常個数
        anomaly_count = widgets.IntSlider(
            value=0, min=0, max=20, step=1,
            description='異常個数:',
            style={'description_width': 'initial'}
        )
        anomaly_points_container = widgets.VBox()
        
        # 予兆個数
        precursor_count = widgets.IntSlider(
            value=0, min=0, max=20, step=1,
            description='予兆個数:',
            style={'description_width': 'initial'}
        )
        precursor_points_container = widgets.VBox()
        
        def update_anomaly_points(change):
            n = change['new'] if isinstance(change, dict) else change
            anomaly_points_container.children = [
                self._create_anomaly_point_widget(i, sensor_id, '異常') for i in range(1, n + 1)
            ]
        
        def update_precursor_points(change):
            n = change['new'] if isinstance(change, dict) else change
            precursor_points_container.children = [
                self._create_anomaly_point_widget(i, sensor_id, '予兆') for i in range(1, n + 1)
            ]
        
        anomaly_count.observe(update_anomaly_points, names='value')
        precursor_count.observe(update_precursor_points, names='value')
        
        return widgets.VBox([
            widgets.HTML(f"<b>センサ{sensor_id} 異常・予兆設定</b>"),
            widgets.HTML("<span style='color:red'>■ 異常設定</span>"),
            anomaly_count,
            anomaly_points_container,
            widgets.HTML("<span style='color:orange'>■ 予兆設定</span>"),
            precursor_count,
            precursor_points_container
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px 0'))
    
    def _update_anomaly_settings_ui(self):
        """異常設定UIを更新"""
        n_sensors = self.n_sensors_input.value
        self.anomaly_settings_container.children = [
            self._create_sensor_anomaly_widget(i) for i in range(1, n_sensors + 1)
        ]
        self.anomaly_settings_container.layout.display = 'block'
        self.apply_anomaly_button.layout.display = 'block'
    
    def _get_anomaly_settings(self) -> tuple[dict, dict]:
        """異常・予兆設定を取得"""
        anomaly_settings = {}
        precursor_settings = {}
        
        for sensor_idx, sensor_box in enumerate(self.anomaly_settings_container.children):
            # sensor_box.children:
            # [0] HTML(センサX), [1] HTML(異常), [2] anomaly_count, [3] anomaly_points,
            # [4] HTML(予兆), [5] precursor_count, [6] precursor_points
            
            anomaly_points_container = sensor_box.children[3]
            precursor_points_container = sensor_box.children[6]
            
            # 異常設定
            anomaly_list = []
            for point_box in anomaly_points_container.children:
                # point_box.children: [0] HTML, [1] Dropdown(日時), [2] FloatText(平均)
                dt = point_box.children[1].value
                mean = point_box.children[2].value
                if dt is not None:
                    anomaly_list.append((dt, mean))
            if anomaly_list:
                anomaly_settings[sensor_idx] = anomaly_list
            
            # 予兆設定
            precursor_list = []
            for point_box in precursor_points_container.children:
                dt = point_box.children[1].value
                mean = point_box.children[2].value
                if dt is not None:
                    precursor_list.append((dt, mean))
            if precursor_list:
                precursor_settings[sensor_idx] = precursor_list
        
        return anomaly_settings, precursor_settings
    
    def _on_generate_normal_click(self, b):
        """正常データ生成ボタンクリック時"""
        with self.output_area:
            clear_output(wait=True)
            try:
                start = datetime.combine(self.start_date_picker.value, datetime.min.time())
                end = datetime.combine(self.end_date_picker.value, datetime.min.time())
                self.interval_hours = self.interval_input.value
                self.sensor_params = self._get_sensor_params()
                
                self.generated_df = generate_dataset(
                    start_date=start,
                    end_date=end,
                    interval_hours=self.interval_hours,
                    sensor_params=self.sensor_params,
                    seed=42
                )
                
                # 日時候補リストを作成
                self.date_options = [(str(dt), dt) for dt in self.generated_df['measurement_date']]
                
                print(f"正常データ生成完了: {len(self.generated_df)}件")
                print(f"データ期間: {self.generated_df['measurement_date'].min()} ~ {self.generated_df['measurement_date'].max()}")
                print(f"取得間隔: {self.interval_hours}時間")
                
                plot_sensor_data(
                    self.generated_df,
                    title="正常センサデータ",
                    y_min=self.y_min_input.value,
                    y_max=self.y_max_input.value
                )
                plt.show()
                
                # 異常設定UIを表示
                self._update_anomaly_settings_ui()
                print("\n異常・予兆設定を行い、[異常データ適用]ボタンを押してください。")
                
            except Exception as e:
                print(f"エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_apply_anomaly_click(self, b):
        """異常データ適用ボタンクリック時"""
        with self.output_area:
            clear_output(wait=True)
            try:
                anomaly_settings, precursor_settings = self._get_anomaly_settings()
                
                # 日時の重複チェック（同一センサ内のみ）
                errors = []
                for sensor_idx, settings in anomaly_settings.items():
                    datetimes = [dt for dt, _ in settings]
                    if len(datetimes) != len(set(datetimes)):
                        errors.append(f"センサ{sensor_idx + 1}: 異常の日時が重複しています")
                
                for sensor_idx, settings in precursor_settings.items():
                    datetimes = [dt for dt, _ in settings]
                    if len(datetimes) != len(set(datetimes)):
                        errors.append(f"センサ{sensor_idx + 1}: 予兆の日時が重複しています")
                
                # 同一センサ内で異常と予兆の重複チェック
                for sensor_idx in set(anomaly_settings.keys()) & set(precursor_settings.keys()):
                    anomaly_dts = {dt for dt, _ in anomaly_settings[sensor_idx]}
                    precursor_dts = {dt for dt, _ in precursor_settings[sensor_idx]}
                    overlap = anomaly_dts & precursor_dts
                    if overlap:
                        errors.append(f"センサ{sensor_idx + 1}: 異常と予兆の日時が重複しています")
                
                if errors:
                    print("エラー:")
                    for e in errors:
                        print(f"  {e}")
                    return
                
                # 正常データを再生成してから異常適用
                start = datetime.combine(self.start_date_picker.value, datetime.min.time())
                end = datetime.combine(self.end_date_picker.value, datetime.min.time())
                
                self.generated_df = generate_dataset(
                    start_date=start,
                    end_date=end,
                    interval_hours=self.interval_hours,
                    sensor_params=self.sensor_params,
                    seed=42
                )
                
                self.generated_df = apply_anomalies(
                    df=self.generated_df,
                    interval_hours=self.interval_hours,
                    sensor_params=self.sensor_params,
                    anomaly_settings=anomaly_settings,
                    precursor_settings=precursor_settings,
                    seed=42
                )
                
                # センサごとの統計表示
                print("データ適用完了:")
                n_sensors = len(self.sensor_params)
                for i in range(1, n_sensors + 1):
                    flag_col = f'sensor{i}_flag'
                    n_normal = (self.generated_df[flag_col] == 0).sum()
                    n_precursor = (self.generated_df[flag_col] == 1).sum()
                    n_anomaly = (self.generated_df[flag_col] == 2).sum()
                    print(f"  sensor{i}: 正常={n_normal}, 予兆={n_precursor}, 異常={n_anomaly}")
                
                plot_sensor_data(
                    self.generated_df,
                    title="センサデータ（異常・予兆適用済み）",
                    y_min=self.y_min_input.value,
                    y_max=self.y_max_input.value
                )
                plt.show()
                
            except Exception as e:
                print(f"エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_save_click(self, b):
        """保存ボタンクリック時"""
        with self.output_area:
            if self.generated_df is None:
                print("先にデータを生成してください")
                return
            try:
                filepath = save_dataset(self.generated_df)
                print(f"保存完了: {filepath}")
            except Exception as e:
                print(f"保存エラー: {e}")
    
    def show(self):
        """UIを表示"""
        display(widgets.VBox([
            widgets.HTML("<h3>共通パラメータ</h3>"),
            widgets.HBox([self.start_date_picker, self.end_date_picker, self.interval_input]),
            widgets.HBox([self.y_min_input, self.y_max_input]),
            widgets.HTML("<h3>センサ設定</h3>"),
            self.n_sensors_input,
            self.sensor_params_container,
            widgets.HBox([self.generate_normal_button, self.apply_anomaly_button, self.save_button]),
            widgets.HTML("<h3>異常・予兆設定</h3>"),
            self.anomaly_settings_container,
            self.output_area
        ]))