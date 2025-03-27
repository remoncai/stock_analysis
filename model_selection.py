import numpy as np
import pandas as pd
import akshare as ak
import os

# 创建保存结果的目录
os.makedirs('results', exist_ok=True)

# 获取股票数据
def get_stock_data(stock_code, market):
    """获取股票历史数据"""
    try:
        if market == 'A':
            # 获取 A 股数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20200101", adjust="qfq")
        elif market == 'HK':
            # 获取港股数据
            df = ak.stock_hk_hist(symbol=stock_code, period="daily", start_date="20200101", adjust="qfq")
        else:
            raise ValueError("无效的市场类型，请输入 'A' 或 'HK'")

        if df is None or df.empty:
            print(f"错误：未能获取到股票 {stock_code} 的数据")
            return None

        # 转换日期格式并设置为索引
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df
    except Exception as e:
        print(f"获取股票 {stock_code} 数据时出错: {str(e)}")
        return None

# 计算历史波动率
def calculate_historical_volatility(df, window=30):
    df['收盘'] = df['收盘'].astype(float)  # 确保数据类型为 float
    df['日收益率'] = df['收盘'].pct_change()
    df['历史波动率'] = df['日收益率'].rolling(window=window).std() * np.sqrt(252)
    return df['历史波动率'].iloc[-1]

# 根据波动性选择模型
def select_model(stock_code, volatility):
    if volatility == '高':
        model_selection = "单层 LSTM"
        print(f"股票 {stock_code} 波动性较高，使用单层 LSTM")
    elif volatility == '中':
        model_selection = "单层 LSTM + 新闻情绪"
        print(f"股票 {stock_code} 波动性中等，使用单层 LSTM + 新闻情绪")
    elif volatility == '低':
        model_selection = "双层 LSTM"
        print(f"股票 {stock_code} 波动性较低，使用双层 LSTM")
    else:
        raise ValueError("波动性参数无效")
    return model_selection

# 保存结果到文件
def save_results_to_file(stock_code, market, historical_volatility, model_selection):
    """将分析结果保存到 .txt 文件"""
    file_path = f"results/{stock_code}_{market}_analysis.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"股票代码: {stock_code}\n")
        f.write(f"市场类型: {'A 股' if market == 'A' else '港股'}\n")
        f.write(f"历史波动率: {historical_volatility:.4f}\n")
        f.write(f"模型选择: {model_selection}\n")
    print(f"分析结果已保存至: {file_path}")

# 主程序
if __name__ == "__main__":
    # 定义股票代码列表和市场类型
    stocks = [
        {"code": "603387", "market": "A"},  # A 股
        {"code": "603259", "market": "A"},  # A 股
        {"code": "600683", "market": "A"},  # A 股
        {"code": "603728", "market": "A"},  # A 股
        {"code": "300122", "market": "A"},  # A 股
        {"code": "300124", "market": "A"},  # A 股
        {"code": "300751", "market": "A"},  # A 股
        {"code": "300896", "market": "A"},  # A 股
        {"code": "300999", "market": "A"},  # A 股
        {"code": "02096", "market": "HK"},  # 港股
        {"code": "09880", "market": "HK"},  # 港股
        {"code": "002459", "market": "A"},  # A股
        {"code": "002096", "market": "A"},  # A股
        {"code": "002232", "market": "A"},  # A股
        {"code": "300493", "market": "A"},  # A股
        {"code": "002031", "market": "A"},  # A股
        {"code": "002028", "market": "A"},  # A股
    ]

    # 创建一个空的 DataFrame 用于存储所有股票的分析结果
    results_df = pd.DataFrame(columns=["股票代码", "市场类型", "历史波动率", "模型选择"])

    # 遍历股票代码列表
    for stock in stocks:
        stock_code = stock["code"]
        market = stock["market"]
        print(f"\n开始分析股票代码: {stock_code} (市场: {'A 股' if market == 'A' else '港股'})")

        # 获取股票数据
        df = get_stock_data(stock_code, market)
        if df is None:
            continue  # 如果获取数据失败，跳过当前股票

        # 计算历史波动率
        historical_volatility = calculate_historical_volatility(df)

        # 根据波动性分类
        if historical_volatility > 0.4:  # 假设波动率 > 40% 为高波动性
            volatility_level = '高'
        elif historical_volatility > 0.2:  # 假设波动率 > 20% 为中等波动性
            volatility_level = '中'
        else:  # 波动率 <= 20% 为低波动性
            volatility_level = '低'

        # 选择模型
        model_selection = select_model(stock_code, volatility_level)
        print(f"股票 {stock_code} 波动性: {volatility_level}, 模型选择: {model_selection}")

        # 将结果添加到 DataFrame 中
        new_row = pd.DataFrame({
            "股票代码": [stock_code],
            "市场类型": ['A 股' if market == 'A' else '港股'],
            "历史波动率": [historical_volatility],
            "模型选择": [model_selection]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 保存所有结果到一个文件
    output_file_path = "results/all_stocks_analysis.csv"  # 保存为 CSV 文件
    results_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"\n所有股票的分析结果已保存至: {output_file_path}")