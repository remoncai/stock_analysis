import akshare as ak
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openpyxl import load_workbook
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Your TensorFlow code here
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(...)

# 创建保存结果的目录
os.makedirs('results', exist_ok=True)

def get_stock_data(stock_code):
    """获取股票历史数据"""
    try:
        # 获取股票日K线数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20200101", adjust="qfq")  # 扩大历史数据范围
        if df is None or df.empty:
            print("错误：未能获取到股票数据")
            return None

        # 转换日期格式并设置为索引
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df

    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None

# 获取新闻情绪数据
def get_news_sentiment(news_texts):
    sentiments = []
    for text in news_texts:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
    return np.mean(sentiments)

def calculate_technical_indicators(df):
    """计算技术指标"""
    df = df.copy()

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

    return df.fillna(method='bfill')

def prepare_sequences(data, seq_length):
    """准备LSTM序列数据"""
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def create_lstm_model(seq_length, n_features):
    """创建LSTM模型"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),  # 增加神经元数量
        Dropout(0.3),  # 增加dropout比例
        LSTM(64, return_sequences=False),  # 增加神经元数量
        Dropout(0.3),
        Dense(32),  # 增加神经元数量
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def prepare_data_for_training(df, seq_length=60):
    """准备训练数据"""
    # 选择特征
    feature_columns = ['收盘', 'MA5', 'MA20', 'MA60', 'RSI', 'MACD',
                      'BB_middle', 'BB_upper', 'BB_lower', '动量',
                      '成交量_变化率', '收盘_涨跌幅']

    # 准备用于缩放的数据
    price_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()

    # 单独对收盘价进行缩放
    prices = df['收盘'].values.reshape(-1, 1)
    scaled_prices = price_scaler.fit_transform(prices)

    # 对特征进行缩放
    features = df[feature_columns].values
    scaled_features = feature_scaler.fit_transform(features)

    # 准备序列数据
    X, y = prepare_sequences(scaled_features, seq_length)
    y = scaled_prices[seq_length:]

    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, X_val, y_train, y_val, price_scaler, feature_scaler, feature_columns

def predict_future_prices(model, last_sequence, price_scaler, feature_scaler, days=5):
    """预测未来价格"""
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        # 预测下一个值
        scaled_pred = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]))

        # 反转标准化并获取价格
        pred_price = price_scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0][0]
        predictions.append(pred_price)

        # 更新序列
        new_seq = np.roll(current_seq, -1, axis=0)
        new_seq[-1] = current_seq[-1]  # 使用最后一个特征集
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
    future_dates = pd.date_range(start=df.index[-1], periods=len(future_pred)+1, freq='B')[1:]
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
            idx = -i-1
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
    formatter.scaled[1/86400] = '%Y-%m-%d'  # 日期格式
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
    image_path = f'results/{stock_code}_{current_time}.png'
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {image_path}")

    plt.close('all')  # 关闭所有图形窗口

    return future_dates

def analyze_stock(stock_code):
    """主分析函数"""
    # 获取股票名称
    try:
        stock_info = ak.stock_info_a_code_name()
        stock_name = stock_info[stock_info['code'] == stock_code]['name'].values[0]
    except:
        stock_name = ""

    print(f"\n开始分析股票 {stock_name}({stock_code}) 的价格趋势...")
    # 获取数据
    df = get_stock_data(stock_code)
    if df is None:
        return

    # 获取新闻情绪数据
    news_texts = ["股票市场暴涨！", "经济不确定性增加."]
    sentiment_score = get_news_sentiment(news_texts)
    df['新闻情绪'] = sentiment_score  # 假设情绪分数适用于整个时间段

    # 计算技术指标
    df = calculate_technical_indicators(df)

    # 准备训练数据
    seq_length = 365  # 使用365天的数据来预测
    X_train, X_val, y_train, y_val, price_scaler, feature_scaler, feature_columns = prepare_data_for_training(df, seq_length)

    # 创建和训练模型
    model = create_lstm_model(seq_length, len(feature_columns))
    print("\n开始训练模型...")
    print("每个epoch代表一轮完整的训练，共50轮：")
    # 定义早停法和学习率调度器
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证损失
        patience=10,  # 等待10轮后停止训练
        restore_best_weights=True  # 恢复最佳权重
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证损失
        factor=0.1,  # 学习率降低因子
        patience=5,  # 等待5轮后降低学习率
        min_lr=1e-6  # 最小学习率
    )
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,  # 增加批次大小以加快训练
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr],  # 添加学习率调度器
        verbose=1,
        shuffle=True
    )

    # 可视化训练过程
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    # 保存图表到文件
    loss_curve_path = f'results/训练和验证损失曲线_{stock_name}({stock_code}).png'
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表，避免阻塞程序运行

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
    mape = np.mean(np.abs((df['收盘'].values[-len(history_pred):] - history_pred) / df['收盘'].values[-len(history_pred):])) * 100

    # 绘制图表并获取未来日期
    future_dates = plot_results(df, history_pred, future_pred, stock_code, stock_name)

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
    csv_path = f'results/{stock_code}_{current_time}.csv'
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
    print(f"新闻情绪:{df['新闻情绪'] .iloc[-1]:.2f} ")
    print(f"\n预测准确度：")
    print(f"均方根误差(RMSE): {rmse:.2f} 元")
    print(f"平均百分比误差(MAPE): {mape:.2f}%")

    print(f"\n未来5个工作日预测价格：")
    for date, price in zip(future_dates, future_pred):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} 元")

    # 计算预测趋势
    pred_trend = (future_pred[-1] - future_pred[0]) / future_pred[0] * 100

    # 输出趋势分析
    latest_ma5 = df['MA5'].iloc[-1]
    latest_ma20 = df['MA20'].iloc[-1]
    latest_price = df['收盘'].iloc[-1]

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

    # 将分析结果导出到 txt 文件
    txt_path = f'results/{stock_code}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"股票代码: {stock_code}\n")
        f.write(f"股票名称: {stock_name}\n")
        f.write(f"均方根误差(RMSE): {rmse:.2f} 元\n")
        f.write(f"平均百分比误差(MAPE): {mape:.2f}%\n")
        f.write(f"2025-03-25: {future_pred[0]:.2f} 元\n")
        f.write(f"2025-03-26: {future_pred[1]:.2f} 元\n")
        f.write(f"2025-03-27: {future_pred[2]:.2f} 元\n")
        f.write(f"2025-03-28: {future_pred[3]:.2f} 元\n")
        f.write(f"2025-03-31: {future_pred[4]:.2f} 元\n")
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

if __name__ == "__main__":
    # 定义 Excel 文件路径 - 使用当前工作目录下的Excel文件
    excel_path = "all_stocks_analysis_0323_demo.xlsx"

    # 定义可能的文件路径列表
    possible_paths = [
        'D:/documents/GitHub/stock_analysis/all_stocks_analysis_0323_demo.xlsx',  # 默认路径
        'D:/bigdata/all_stocks_analysis_0323_demo.xlsx',  # 替代路径
    ]

    # 动态查找文件
    excel_file = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_file = path
            break

    if excel_file:
        print(f"找到文件: {excel_file}")
        df = pd.read_excel(excel_file, sheet_name=0)  # 读取第一个工作表
    else:
        print("错误：未找到文件，请确认文件路径是否正确")

    try:  # 添加 try 块以修复缩进错误
        # 读取股票代码（A 列，从第 3 行开始）
        stock_codes = []
        for row in ws.iter_rows(min_row=3, max_col=1, values_only=True):
            stock_code = row[0]
            if stock_code:
                stock_codes.append(stock_code)

        # 打印读取的股票代码
        print(f"读取到的股票代码: {stock_codes}")

        # 继续处理股票代码
        if stock_codes:
            print("开始处理股票代码...")
            # 在这里添加处理股票代码的逻辑
        else:
            print("未读取到有效的股票代码，请检查 Excel 文件内容。")

    except Exception as e:
        print(f"加载 Excel 文件时出错: {str(e)}")
    else:
        print("错误：未找到文件，请确认文件路径是否正确")

    # 打印当前工作目录，帮助调试
    import os
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试读取文件: {excel_path}")

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"文件不存在! 请确认文件路径是否正确。")
        # 尝试在其他可能的位置查找
        alternative_path = r"D:\bigdata\all_stocks_analysis_0323_demo.xlsx"
        if os.path.exists(alternative_path):
            print(f"找到替代文件位置: {alternative_path}")
            excel_path = alternative_path
        else:
            print(f"替代位置文件也不存在: {alternative_path}")

    # 使用 pandas 读取 Excel 文件
    try:
        # 读取 Excel 文件中的第一个工作表
        df = pd.read_excel(excel_path, sheet_name=0)  # sheet_name=0 表示第一个工作表

        # 打印Excel文件的基本信息
        print(f"Excel文件读取成功，共有 {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"前5行数据预览:")
        print(df.head())

        # 打印C列(第2列)的唯一值，帮助确认筛选条件
        unique_values = df.iloc[:, 4].unique()
        print(f"C列(第2列)的唯一值: {unique_values}")

        # 筛选出C列（第2列）为"单层 LSTM + 新闻情绪"的行
        filtered_df = df[df.iloc[:, 4] == "单层 LSTM + 新闻情绪"]
        print(f"筛选后的行数: {filtered_df.shape[0]}")

        # 从筛选后的数据中提取A列（第0列）的股票代码
        stock_codes = filtered_df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"筛选出的股票代码: {stock_codes}")

        # 打印筛选后的数据
        print("筛选后的数据：")
        print(filtered_df)

        # 打印筛选出的股票代码
        print("筛选出的股票代码：")
        print(stock_codes)

    except Exception as e:
        print(f"读取 Excel 文件时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误信息
        stock_codes = []  # 如果出错，设置为空列表

    # 如果成功读取到股票代码，开始分析
    if stock_codes:
        print(f"从 Excel 文件中读取到以下股票代码: {stock_codes}")
        # 遍历股票代码列表，依次分析每个股票
        for stock_code in stock_codes:
            print(f"\n开始分析股票代码: {stock_code}")
            analyze_stock(stock_code)
    else:
        print("未读取到有效的股票代码，请检查 Excel 文件路径和内容。")

    from tqdm import tqdm
    import time

    for i in tqdm(range(100)):
        time.sleep(0.1)  # 模拟任务