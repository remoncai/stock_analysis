# 本程序既可以分析A股又可以分析港股。

import akshare as ak
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import warnings
import os
import re
import sys

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建保存结果的目录
os.makedirs('results/informer', exist_ok=True)

def is_hk_stock(stock_code):
    """判断是否为港股代码"""
    # 港股代码通常是5位数字，以0开头
    return bool(re.match(r'^0\d{4}$', str(stock_code)))

def get_stock_data(stock_code):
    """获取股票历史数据"""
    try:
        if is_hk_stock(stock_code):
            # 获取港股日K线数据
            df = ak.stock_hk_hist(symbol=stock_code, period="daily", start_date="20200101", adjust="")
            if df is None or df.empty:
                print("错误：未能获取到港股数据")
                return None

            # 港股数据列名已经包含'日期'，直接使用
            if '日期' not in df.columns:
                print("错误：港股数据中未找到'日期'列")
                return None

            # 转换日期格式并设置为索引
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)

            # 统一列名（港股数据列名已正确，无需重命名）
            # 只需确保有我们需要的列
            required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
            for col in required_columns:
                if col not in df.columns:
                    print(f"错误：缺少必要列 {col}")
                    return None

        else:
            # 获取A股日K线数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20200101", adjust="qfq")
            if df is None or df.empty:
                print("错误：未能获取到A股数据")
                return None

            # 转换日期格式并设置为索引
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)

            # 检查A股数据列
            required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
            for col in required_columns:
                if col not in df.columns:
                    print(f"错误：缺少必要列 {col}")
                    return None

        return df

    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None

def get_stock_name(stock_code):
    """获取股票名称"""
    try:
        if is_hk_stock(stock_code):
            # 获取港股名称
            hk_stock_list = ak.stock_hk_spot()
            # 确保代码列是字符串类型，并保持完整的5位代码
            hk_stock_list['代码'] = hk_stock_list['代码'].astype(str).str.zfill(5)
            stock_name = hk_stock_list[hk_stock_list['代码'] == str(stock_code).zfill(5)]['名称'].values[0]
        else:
            # 获取A股名称
            stock_info = ak.stock_info_a_code_name()
            stock_name = stock_info[stock_info['code'] == stock_code]['name'].values[0]
        return stock_name
    except Exception as e:
        print(f"获取股票名称时出错: {str(e)}")
        return ""
    except IndexError:
        print(f"无法找到股票代码: {stock_code}")
        return ""

def calculate_technical_indicators(df):
    """计算技术指标"""

    try:
        df = df.copy()

        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')

        # 移动平均线
        for window in [5, 10, 20, 30, 60]:
            df[f'MA{window}'] = df['收盘'].rolling(window=window).mean()
            df[f'成交量MA{window}'] = df['成交量'].rolling(window=window).mean()

        # RSI
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # 布林带
        df['BB_middle'] = df['收盘'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['收盘'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['收盘'].rolling(window=20).std()

        # 价格动量
        df['动量'] = df['收盘'].pct_change()
        df['收盘_涨跌幅'] = df['收盘'].pct_change()
        df['成交量_变化率'] = df['成交量'].pct_change()

        # 最终填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')

        # 确保所有必要特征都存在
        required_features = ['MA5', 'MA20', 'MA60', 'RSI', 'MACD',
                             'BB_middle', 'BB_upper', 'BB_lower', '动量',
                             '收盘_涨跌幅', '成交量_变化率']

        for feature in required_features:
            if feature not in df.columns:
                print(f"警告：未能计算出特征 {feature}，使用0填充")
                df[feature] = 0

        return df

    except Exception as e:
        print(f"计算技术指标时出错: {str(e)}")
        return None

class Time2Vector(tf.keras.layers.Layer):
    """时间嵌入层"""

    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)  # 取前4个特征(开盘、收盘、最高、最低)计算时间嵌入
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)

        return tf.concat([time_linear, time_periodic], axis=-1)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dense(d_model, kernel_regularizer=regularizers.l2(0.001))
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):  # 添加默认参数training=False
        attn_output = self.mha(x, x, x)  # 自注意力
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

def create_informer_model(seq_length, n_features, d_model=64, num_heads=4, dff=256, rate=0.2):
    """创建的Informer模型（兼容TensorFlow 2.19.0和Keras 3.9.0）
    参数调整说明：
    - d_model: 从128减少到64，降低模型复杂度
    - num_heads: 从8减少到4，减少注意力机制的复杂度
    - dff: 从512减少到256，减少前馈网络的参数量
    - rate: 从0.1增加到0.2，加强dropout正则化
    """
    inputs = Input(shape=(seq_length, n_features))

    # 时间嵌入层
    time_embedding = Time2Vector(seq_length)(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)([inputs, time_embedding])

    # 编码器部分（增加L2正则化）
    x = Dense(d_model, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 减少为2层Transformer编码器
    for _ in range(2):
        x = TransformerEncoderLayer(d_model, num_heads, dff, rate)(x)

    # 注意力池化层
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = tf.keras.layers.Concatenate()([x, attention])

    # 解码器部分（增加L2正则化）
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(dff, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(rate)(x)
    x = Dense(dff // 2, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    # 优化器配置（兼容Keras 3.x）
    optimizer = Adam(
        learning_rate=0.0001,
        global_clipnorm=1.0  # 使用全局梯度裁剪
    )

    # 指标配置（兼容Keras 3.x）
    try:
        # 尝试Keras 3.x的新API
        mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
    except TypeError:
        # 回退到兼容模式
        mape_metric = 'mean_absolute_percentage_error'

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[mape_metric]
    )

    # 调试信息
    print("\n模型验证:")
    print("模型输入形状:", model.input_shape)
    print("模型输出形状:", model.output_shape)
    model.summary()

    return model

def prepare_sequences(data, seq_length):
    """准备序列数据"""
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length, 0]  # 假设收盘价是第一列
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def prepare_data_for_training(df, seq_length=60):
    """准备训练数据"""
    try:
        # 选择特征
        feature_columns = ['收盘', 'MA5', 'MA20', 'MA60', 'RSI', 'MACD',
                          'BB_middle', 'BB_upper', 'BB_lower', '动量',
                          '成交量_变化率', '收盘_涨跌幅']

        # 添加更多特征
        df['价格波动率'] = df['收盘'].pct_change().rolling(5).std()
        df['量价背离'] = (df['成交量'] - df['成交量'].rolling(5).mean()) * (df['收盘'] - df['收盘'].rolling(5).mean())
        feature_columns.extend(['价格波动率', '量价背离'])

        # 检查特征列是否存在
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"错误：缺失必要列 {missing_cols}")
            return None, None, None, None, None, None, None

        # 使用更稳健的缩放方法 - RobustScaler
        from sklearn.preprocessing import RobustScaler
        price_scaler = RobustScaler(quantile_range=(5, 95))
        feature_scaler = RobustScaler(quantile_range=(5, 95))

        # 单独对收盘价进行缩放
        prices = df['收盘'].values.reshape(-1, 1)
        scaled_prices = price_scaler.fit_transform(prices)

        # 对特征进行缩放
        features = df[feature_columns].values  # 确保这行在price_scaler之后
        scaled_features = feature_scaler.fit_transform(features)

        # 处理NaN值 - 填充或删除
        if np.isnan(features).any():
            print("发现NaN值，尝试填充...")
            # 先用前向填充
            df[feature_columns] = df[feature_columns].fillna(method='ffill')
            # 再用后向填充
            df[feature_columns] = df[feature_columns].fillna(method='bfill')
            # 最后用0填充剩余NaN
            df[feature_columns] = df[feature_columns].fillna(0)
            features = df[feature_columns].values

        scaled_features = feature_scaler.fit_transform(features)

        # 再次检查NaN值
        if np.isnan(scaled_features).any():
            print("错误：特征数据中存在NaN值！")
            print("NaN值位置：", np.where(np.isnan(scaled_features)))
            return None, None, None, None, None, None, None

    except Exception as e:
        print(f"准备训练数据时出错: {str(e)}")
        return None, None, None, None, None, None, None

    if np.isnan(scaled_prices).any():
        print("错误：价格数据中存在NaN值！")
        return None, None, None, None, None, None, None

    print(f"使用的特征列: {feature_columns}")

    # 准备序列数据
    X, y = prepare_sequences(scaled_features, seq_length)
    y = scaled_prices[seq_length:]

    print(f"生成的序列形状: X{X.shape}, y{y.shape}")

    if len(X) == 0:
        print("错误：生成的序列数量为0")
        return None, None, None, None, None, None, None

    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, X_val, y_train, y_val, price_scaler, feature_scaler, feature_columns

def predict_future_prices(model, last_sequence, price_scaler, feature_scaler, days=5):
    """预测未来价格"""
    predictions = []
    current_seq = last_sequence.copy()

    # 使用蒙特卡洛dropout进行不确定性估计
    num_samples = 10
    for _ in range(days):
        # 多次预测取平均
        preds = []
        for _ in range(num_samples):
            scaled_pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)
            preds.append(scaled_pred[0, 0])

        # 使用中位数而不是平均值
        scaled_pred = np.median(preds)

        # 反转标准化
        pred_price = price_scaler.inverse_transform([[scaled_pred]])[0][0]
        predictions.append(pred_price)

        # 更智能的序列更新
        new_seq = np.roll(current_seq, -1, axis=0)
        new_seq[-1] = current_seq[-1] * 0.9 + current_seq[-2] * 0.1  # 加权更新
        current_seq = new_seq

    return predictions

def plot_results(df, history_pred, future_pred, stock_code, stock_name):
    """绘制分析图表并保存"""
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # 绘制历史价格
    ax1.plot(df.index, df['收盘'], label='历史价格', color='blue', linewidth=2)

    # 绘制回归价格
    regression_dates = df.index[-len(history_pred):]
    ax1.plot(regression_dates, history_pred,
             label='模型拟合', color='green', alpha=0.7)

    # 添加未来预测 - 使用工作日
    future_dates = pd.date_range(start=df.index[-1], periods=len(future_pred) + 1, freq='B')[1:]
    # 只保留需要的天数
    future_dates = future_dates[:len(future_pred)]

    # 绘制预测价格
    ax1.plot(future_dates, future_pred, label='预测价格',
             color='red', linestyle='--', linewidth=2)

    # 用红色虚线连接最后一个回归价格和第一个预测价格
    connection_dates = [regression_dates[-1], future_dates[0]]
    connection_prices = [history_pred[-1], future_pred[0]]
    ax1.plot(connection_dates, connection_prices, color='red', linestyle='--', linewidth=2)

    # 标记未来预测的点
    for date, price in zip(future_dates, future_pred):
        ax1.scatter(date, price, color='red', s=100, zorder=5)
        ax1.annotate(f'{price:.2f}',
                     xy=(date, price),
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

    # 添加技术指标
    ax1.plot(df.index, df['MA20'], label='20日均线', color='purple', alpha=0.5)
    ax1.plot(df.index, df['BB_upper'], label='布林带上轨', color='gray', alpha=0.3)
    ax1.plot(df.index, df['BB_lower'], label='布林带下轨', color='gray', alpha=0.3)

    # 标注最近5个实际价格和拟合价格
    for i in range(5):
        if len(df) > i and len(history_pred) > i:
            # 获取倒数第i+1个日期和价格
            idx = -i - 1
            actual_date = df.index[idx]
            actual_price = df['收盘'].iloc[idx]
            fitted_idx = len(history_pred) + idx
            if fitted_idx >= 0:
                fitted_price = history_pred[fitted_idx]

                # 标注实际价格
                ax1.scatter(actual_date, actual_price, color='blue', s=80, zorder=5)
                ax1.annotate(f'{actual_price:.2f}',
                             xy=(actual_date, actual_price),
                             xytext=(10, -15), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='blue', alpha=0.8))

                # 标注拟合价格
                ax1.scatter(actual_date, fitted_price, color='green', s=80, zorder=5)
                ax1.annotate(f'{fitted_price:.2f}',
                             xy=(actual_date, fitted_price),
                             xytext=(10, 10), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='green', alpha=0.8))

    # 设置x轴刻度
    from matplotlib.dates import AutoDateLocator, AutoDateFormatter

    # 使用自动日期定位器，但设置最大刻度数量
    locator = AutoDateLocator(maxticks=20)
    ax1.xaxis.set_major_locator(locator)

    # 格式化x轴日期标签
    formatter = AutoDateFormatter(locator)
    formatter.scaled[1 / 86400] = '%Y-%m-%d'  # 日期格式
    ax1.xaxis.set_major_formatter(formatter)

    # 旋转日期标签，使其更易读
    plt.xticks(rotation=45)

    # 设置缩放限制，防止过度放大
    from matplotlib.ticker import MaxNLocator

    # 定义缩放事件处理函数
    def on_xlims_change(event_ax):
        # 获取当前x轴范围
        x_min, x_max = event_ax.get_xlim()

        # 计算当前视图中的天数
        days_in_view = (plt.matplotlib.dates.num2date(x_max) -
                        plt.matplotlib.dates.num2date(x_min)).days

        # 如果放大到少于10天，则限制放大
        if days_in_view < 10:
            # 恢复到10天的视图
            mid_point = (x_min + x_max) / 2
            half_range = 5 * 1.0  # 5天，转换为matplotlib日期单位
            event_ax.set_xlim(mid_point - half_range, mid_point + half_range)

        # 根据当前视图中的天数调整刻度数量
        if days_in_view < 30:
            # 每天一个刻度
            event_ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator())
        else:
            # 自动调整，但限制最大刻度数
            event_ax.xaxis.set_major_locator(AutoDateLocator(maxticks=20))

        # 更新图表
        fig.canvas.draw_idle()

    # 连接缩放事件
    ax1.callbacks.connect('xlim_changed', on_xlims_change)

    title = f'{stock_name}({stock_code}) 股票价格趋势分析'

    ax1.set_title(title, fontsize=14, pad=20)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()

    # 保存图表到文件
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f'results/informer/{stock_code}_{current_time}.png'
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {image_path}")

    plt.close('all')  # 关闭所有图形窗口

    return future_dates

def analyze_stock(stock_code):
    """主分析函数"""
    # 添加版本检查
    print(f"\n环境信息:")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"Keras版本: {tf.keras.__version__}")

    # 初始化所有可能用到的变量
    history = None
    X_train = None
    y_train = None
    df = None
    stock_name = ""

    try:
        # 获取股票名称
        stock_name = get_stock_name(stock_code)
        print(f"\n开始分析股票 {stock_name}({stock_code}) 的价格趋势...")

        # 获取数据
        df = get_stock_data(stock_code)
        if df is None:
            print("错误：无法获取股票数据")
            return

        # 根据数据长度动态调整序列长度
        seq_length = min(60, len(df) // 4)
        if seq_length < 30:
            print(f"错误：数据长度不足({len(df)})，无法进行分析")
            return

        print(f"使用序列长度: {seq_length}")

        # 计算技术指标
        df = calculate_technical_indicators(df)
        if df is None:
            print("错误：无法计算技术指标")
            return

        # 准备训练数据
        result = prepare_data_for_training(df, seq_length)
        if result is None:
            print("错误：无法准备训练数据")
            return

        X_train, X_val, y_train, y_val, price_scaler, feature_scaler, feature_columns = result

        # 检查数据是否有效
        if X_train is None or len(X_train) == 0:
            print("错误：训练数据无效")
            return

        # 动态批处理大小
        batch_size = min(256, len(X_train))

        # 定义早停法和学习率调度器
        early_stopping = EarlyStopping(
            monitor='val_loss',  # 监控MAPE而不是loss
            patience=30,
            mode='min',
            restore_best_weights=True,
            min_delta=0.001
        )

        # 增加epoch数量
        epochs = 200

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )

        # 修改ModelCheckpoint回调（训练中自动保存最佳模型）
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'best_model_{stock_code}.keras',
            save_weights_only=False,  # 保存完整模型而不是仅权重
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        csv_logger = tf.keras.callbacks.CSVLogger(f'training_log_{stock_code}.csv', append=True, separator=';')

        # +++ 添加TensorBoard回调 +++
        tensorboard = TensorBoard(
            log_dir=f'logs/{stock_code}',
            histogram_freq=1,
            write_graph=True
        )

        callbacks = [early_stopping, reduce_lr, checkpoint, csv_logger, tensorboard]

        # 创建模型
        n_features = len(feature_columns)
        model = create_informer_model(seq_length, n_features)

        #训练前添加数据验证：
        print("\n=== 数据验证 ===")
        # 1. 更严格的数据检查
        print("训练数据有效性检查:")
        print(f"X_train NaN: {np.isnan(X_train).any()}, Inf: {np.isinf(X_train).any()}")
        print(f"y_train NaN: {np.isnan(y_train).any()}, Inf: {np.isinf(y_train).any()}")
        print(f"X_val NaN: {np.isnan(X_val).any()}, Inf: {np.isinf(X_val).any()}")
        print(f"y_val NaN: {np.isnan(y_val).any()}, Inf: {np.isinf(y_val).any()}")

        # 2. 更全面的统计信息
        print("\n训练数据统计:")
        print(f"X_train shape: {X_train.shape}, 范围: [{np.nanmin(X_train):.4f}, {np.nanmax(X_train):.4f}]")
        print(f"y_train shape: {y_train.shape}, 范围: [{np.nanmin(y_train):.4f}, {np.nanmax(y_train):.4f}]")

        # 3. 代表性样本展示
        print("\n特征示例 (第一条序列的最后5个时间步):")
        sample_df = pd.DataFrame(X_train[0][-5:], columns=feature_columns)
        print(sample_df)

        print("\n特征相关性矩阵:")
        try:
            # 计算第一条序列的特征相关性
            corr_matrix = pd.DataFrame(X_train[0], columns=feature_columns).corr()

            # 使用Styler对象增强显示（需要运行在支持HTML的环境中如Jupyter）
            if 'IPython' in sys.modules:
                display(corr_matrix.style.background_gradient(cmap='coolwarm'))
            else:
                print(corr_matrix.round(2))  # 普通终端环境简化输出

            # 保存相关性矩阵图片
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('特征相关性热图')
            plt.savefig(f'results/informer/{stock_code}_correlation.png')
            plt.close()
        except Exception as e:
            print(f"生成相关性矩阵失败: {str(e)}")

        # 4. 安全的初始预测检查
        try:
            print("\n执行初始前向传播验证...")
            with tf.device('/cpu:0'):  # 避免GPU相关错误
                test_output = model.predict(X_train[:1].astype(np.float32))  # 确保数据类型

            if np.isnan(test_output).any() or np.isinf(test_output).any():
                raise ValueError("初始预测包含无效值(nan/inf)!")

            print(f"初始预测输出: {test_output[0, 0]:.4f} (应接近y_train均值: {np.nanmean(y_train):.4f})")
        except Exception as e:
            print(f"初始验证失败: {str(e)}")
            print("可能原因:")
            print("- 模型初始化问题")
            print("- 输入数据包含异常值")
            print("- GPU计算错误(尝试设置TF_CPP_MIN_LOG_LEVEL=2)")

            # 尝试使用CPU重新初始化
            print("\n尝试使用CPU重新初始化...")
            with tf.device('/cpu:0'):
                model = create_informer_model(seq_length, n_features)
                test_output = model.predict(X_train[:1])
                print(f"CPU模式预测: {test_output[0, 0]:.4f}")

        # 训练模型
        print("\n开始训练模型...")
        history = None
        try:
            # 首次训练尝试
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, checkpoint, csv_logger],
                verbose=1,
                shuffle=True
            )

            # 验证训练结果
            test_pred = model.predict(X_train[:1])
            if np.isnan(test_pred).any():
                raise ValueError("模型输出包含NaN值，训练失败！")

        except Exception as e:
            print(f"\n训练过程中出现严重错误: {str(e)}")
            print("输入数据检查:")
            print(f"X_train范围: [{np.nanmin(X_train):.4f}, {np.nanmax(X_train):.4f}]")
            print(f"y_train范围: [{np.nanmin(y_train):.4f}, {np.nanmax(y_train):.4f}]")

            print("\n=== 使用简化模型重试 ===")
            try:
                # 正确的简化模型定义
                simple_model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(64, input_shape=(seq_length, n_features)),
                    tf.keras.layers.Dense(1)
                ])
                simple_model.compile(optimizer=Adam(0.0001), loss='mae')

                # 简化模型训练
                history = simple_model.fit(
                    X_train, y_train,
                    epochs=min(50, epochs),  # 减少epoch数量
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1
                )
                model = simple_model
            except Exception as fallback_e:
                print(f"简化模型也失败: {str(fallback_e)}")
                return None

        # 仅在训练成功时加载模型
        if os.path.exists(f'best_model_{stock_code}.h5'):
            try:
                model.load_weights(f'best_model_{stock_code}.h5')
            except Exception as e:
                print(f"加载权重失败: {str(e)}，使用初始化参数")
        else:
            print("警告：未找到保存的最佳模型权重，使用初始化参数")

        # 当使用简化模型时，修改保存路径
        # 在简化模型训练代码块中：
        simple_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'simple_best_model_{stock_code}.keras',
            save_weights_only=False,  # 保存完整模型而不是仅权重
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

    except Exception as e:
        print(f"分析股票时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return  # 直接返回，避免访问未初始化的变量

    # 可视化训练过程
    # 只在变量已初始化时才执行后续代码
    if X_train is not None and history is not None:
        # 可视化训练过程
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失曲线')
        plt.legend()

        # 保存图表到文件
        loss_curve_path = f'results/informer/训练和验证损失曲线_{stock_name}({stock_code}).png'
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.hist(X_train.flatten(), bins=50)
        plt.title('X_train分布')
        plt.subplot(122)
        plt.hist(y_train.flatten(), bins=50)
        plt.title('y_train分布')
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练和验证损失曲线已保存至: {loss_curve_path}")

    # 获取模型在历史数据上的预测
    print("生成历史数据预测...")
    scaled_features = feature_scaler.transform(df[feature_columns].values)
    sequences, _ = prepare_sequences(scaled_features, seq_length)
    history_pred_scaled = model.predict(sequences, verbose=0)
    history_pred = price_scaler.inverse_transform(history_pred_scaled).flatten()

    # 预测未来价格
    print("预测未来价格趋势...")
    last_sequence = sequences[-1]
    future_pred = predict_future_prices(model, last_sequence, price_scaler, feature_scaler, days=5)

    # 计算预测准确度
    mse = np.mean((df['收盘'].values[-len(history_pred):] - history_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(
        np.abs((df['收盘'].values[-len(history_pred):] - history_pred) / df['收盘'].values[-len(history_pred):])) * 100

    # 计算预测趋势
    pred_trend = (future_pred[-1] - future_pred[0]) / future_pred[0] * 100

    # 获取最新指标
    latest_ma5 = df['MA5'].iloc[-1]
    latest_ma20 = df['MA20'].iloc[-1]
    latest_price = df['收盘'].iloc[-1]

    # 绘制图表后调用error_analysis
    future_dates = plot_results(df, history_pred, future_pred, stock_code, stock_name)
    error_analysis(
        df=df,
        history_pred=history_pred,
        stock_code=stock_code,
        stock_name=stock_name,
        future_dates=future_dates,  # 新增参数
        future_pred=future_pred,
        pred_trend=pred_trend,
        rmse=rmse,
        mape=mape,
        latest_price=latest_price,
        latest_ma5=latest_ma5,
        latest_ma20=latest_ma20
                       )

    # 保存数据到CSV文件
    # 创建历史价格和回归价格的DataFrame
    history_data = pd.DataFrame({
        '日期': df.index[-len(history_pred):],
        '历史价格': df['收盘'].values[-len(history_pred):],
        '回归价格': history_pred
    })

    # 创建预测价格的DataFrame，对应的实际价格为空
    future_data = pd.DataFrame({
        '日期': future_dates,
        '历史价格': np.nan,  # 实际价格为空
        '回归价格': future_pred  # 预测价格放在回归价格列
    })

    # 合并数据
    all_data = pd.concat([history_data, future_data])

    # 保存到CSV
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/informer/{stock_code}_{current_time}.csv'
    all_data.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"数据已保存至: {csv_path}")

    # 输出技术指标
    print("\n技术指标分析：")
    print(f"当前 MA5: {df['MA5'].iloc[-1]:.2f} 元")
    print(f"当前 MA20: {df['MA20'].iloc[-1]:.2f} 元")
    print(f"当前 MA60: {df['MA60'].iloc[-1]:.2f} 元")
    print(f"当前 RSI: {df['RSI'].iloc[-1]:.2f}")
    print(f"当前 MACD: {df['MACD'].iloc[-1]:.2f}")
    print(f"当前布林带上轨: {df['BB_upper'].iloc[-1]:.2f} 元")
    print(f"当前布林带中轨: {df['BB_middle'].iloc[-1]:.2f} 元")
    print(f"当前布林带下轨: {df['BB_lower'].iloc[-1]:.2f} 元")

    # 打印分析结果
    print("\n分析结果：")
    print(f"最新收盘价: {df['收盘'].iloc[-1]:.2f} 元")
    print(f"\n预测准确度：")
    print(f"均方根误差(RMSE): {rmse:.2f} 元")
    print(f"平均百分比误差(MAPE): {mape:.2f}%")

    print(f"\n未来5个工作日预测价格：")
    for date, price in zip(future_dates, future_pred):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} 元")

    print("\n趋势分析：")
    print(f"预测5日涨跌幅：{pred_trend:.2f}%")
    if latest_price > latest_ma5 > latest_ma20:
        print("当前趋势：短期和中期趋势向上")
    elif latest_price < latest_ma5 < latest_ma20:
        print("当前趋势：短期和中期趋势向下")
    elif latest_price > latest_ma5 and latest_ma5 < latest_ma20:
        print("当前趋势：短期趋势转折，需要观察")
    else:
        print("当前趋势：趋势不明确，建议观望")

    if pred_trend > 3:
        print("预测趋势：未来5日可能上涨")
    elif pred_trend < -3:
        print("预测趋势：未来5日可能下跌")
    else:
        print("预测趋势：未来5日可能震荡")

def error_analysis(df, history_pred, stock_code, stock_name, future_dates, future_pred, pred_trend, rmse, mape, latest_price, latest_ma5, latest_ma20):
    """分析误差模式"""
    try:
        errors = df['收盘'].values[-len(history_pred):] - history_pred

        # 检查误差是否有效
        if np.isnan(errors).any() or np.isinf(errors).any():
            print("警告：误差数据包含无效值，跳过分析")
            return

        # 绘制误差分布
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50)
        plt.title('误差分布')
        plt.savefig('error_distribution.png')

        # 分析误差与特征的关系
        error_df = pd.DataFrame({
            'error': errors,
            'RSI': df['RSI'].values[-len(history_pred):],
            'volume': df['成交量'].values[-len(history_pred):]
        })
        print(error_df.corr())

        # 将分析结果导出到 txt 文件
        txt_path = f'results/informer/{stock_code}.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"股票代码: {stock_code}\n")
            f.write(f"股票名称: {stock_name}\n")
            f.write(f"均方根误差(RMSE): {rmse:.2f} 元\n")
            f.write(f"平均百分比误差(MAPE): {mape:.2f}%\n")
            # 写入未来预测日期和价格
            for i, (date, price) in enumerate(zip(future_dates, future_pred)):
                f.write(f"{date.strftime('%Y-%m-%d')}: {price:.2f} 元\n")
            f.write(f"预测5日涨跌幅: {pred_trend:.2f}%\n")
            if latest_price > latest_ma5 > latest_ma20:
                f.write("当前趋势: 短期和中期趋势向上\n")
            elif latest_price < latest_ma5 < latest_ma20:
                f.write("当前趋势: 短期和中期趋势向下\n")
            elif latest_price > latest_ma5 and latest_ma5 < latest_ma20:
                f.write("当前趋势: 短期趋势转折，需要观察\n")
            else:
                f.write("当前趋势: 趋势不明确，建议观望\n")
            if pred_trend > 3:
                f.write("预测趋势: 未来5日可能上涨\n")
            elif pred_trend < -3:
                f.write("预测趋势: 未来5日可能下跌\n")
            else:
                f.write("预测趋势: 未来5日可能震荡\n")

        print(f"\n分析结果已保存至: {txt_path}")

    except Exception as e:
        print(f"误差分析失败: {str(e)}")

if __name__ == "__main__":
    stock_code = input("Enter stock code (A股代码如600519，港股代码如02096): ")
    # 如果用户没有输入，使用默认值600519（贵州茅台）
    if not stock_code.strip():
        stock_code = "600519"
        print(f"使用默认股票代码：{stock_code}（贵州茅台）")

    analyze_stock(stock_code)

    from tqdm import tqdm
    import time

    for i in tqdm(range(100)):
        time.sleep(0.1)  # 模拟任务