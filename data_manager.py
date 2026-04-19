"""
数据管理模块 - 统一管理量化数据的下载、缓存和加载
支持从 baostock 下载A股数据并缓存到本地CSV文件
"""
import os
import pandas as pd
import baostock as bs
from datetime import datetime
from pathlib import Path

# 数据存储目录
DATA_DIR = Path(__file__).parent / 'data'

def ensure_data_dir():
    """确保数据目录存在"""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR

def get_local_filename(symbol, start_date, end_date):
    """生成本地数据文件名
    
    Args:
        symbol: 股票代码 (如 'sh.601138' 或 '601138')
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        文件名 (不含路径)
    """
    # 标准化股票代码
    if '.' not in symbol:
        symbol = 'sh.' + symbol if symbol.startswith('6') else 'sz.' + symbol
    
    # 提取纯代码作为文件名
    code = symbol.split('.')[1]
    return f"{code}.csv"

def download_stock_data(symbol, start_date='2020-01-01', end_date=None, 
                        fields='date,code,open,high,low,close,volume,amount,turn,pctChg',
                        adjustflag='2', force=False):
    """
    下载股票历史数据并保存到本地
    
    Args:
        symbol: 股票代码 (如 'sh.601138' 或 '601138')
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期，默认今天
        fields: 获取的字段
        adjustflag: 复权类型，'2'为前复权
        force: 是否强制重新下载
    
    Returns:
        DataFrame 包含OHLCV数据
    """
    ensure_data_dir()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 标准化股票代码
    if '.' not in symbol:
        symbol = 'sh.' + symbol if symbol.startswith('6') else 'sz.' + symbol
    
    # 本地文件路径
    filename = get_local_filename(symbol, start_date, end_date)
    filepath = DATA_DIR / filename
    
    # 如果本地文件存在且不强制更新，检查是否需要更新
    if filepath.exists() and not force:
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        # 检查日期范围
        last_date = df.index[-1].strftime('%Y-%m-%d')
        if last_date >= end_date or (datetime.now() - datetime.strptime(last_date, '%Y-%m-%d')).days < 1:
            print(f"使用本地缓存数据: {filename} ({len(df)} 条记录)")
            return df
        print(f"本地数据需要更新: 最后日期 {last_date}")
    
    print(f"从 baostock 下载: {symbol} ({start_date} ~ {end_date})")
    
    # 登录 baostock
    bs.login()
    
    # 查询数据
    rs = bs.query_history_k_data_plus(
        symbol,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency='d',
        adjustflag=adjustflag
    )
    
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    
    bs.logout()
    
    if not data:
        print(f"警告: 未获取到数据 {symbol}")
        return None
    
    # 创建DataFrame
    columns = fields.split(',')
    df = pd.DataFrame(data, columns=columns)
    
    # 类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 设置日期索引
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # 保存到本地
    df.to_csv(filepath)
    print(f"已保存到: {filepath} ({len(df)} 条记录)")
    
    return df

def load_stock_data(symbol, start_date='2020-01-01', end_date=None, auto_download=True, **kwargs):
    """
    加载股票数据（优先从本地加载，不存在则下载）
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        auto_download: 如果本地没有数据，是否自动下载
        **kwargs: 传递给 download_stock_data 的参数
    
    Returns:
        DataFrame 或 None
    """
    ensure_data_dir()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 标准化股票代码
    if '.' not in symbol:
        symbol = 'sh.' + symbol if symbol.startswith('6') else 'sz.' + symbol
    
    # 查找本地文件
    filename = get_local_filename(symbol, start_date, end_date)
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        # 过滤日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        print(f"加载本地数据: {filename} ({len(df)} 条记录)")
        return df
    
    if auto_download:
        return download_stock_data(symbol, start_date, end_date, **kwargs)
    
    print(f"本地无数据且未启用自动下载: {symbol}")
    return None

def download_multiple(symbols, start_date='2020-01-01', end_date=None, **kwargs):
    """
    批量下载多只股票数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        dict: {symbol: DataFrame}
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    results = {}
    for symbol in symbols:
        try:
            df = download_stock_data(symbol, start_date, end_date, **kwargs)
            if df is not None:
                results[symbol] = df
        except Exception as e:
            print(f"下载失败 {symbol}: {e}")
    
    return results

def download_dividend_data(symbol, start_year='2020', end_year=None, force=False):
    """
    下载股票分红数据
    
    Args:
        symbol: 股票代码
        start_year: 开始年份
        end_year: 结束年份
        force: 是否强制重新下载
    
    Returns:
        DataFrame 包含分红信息
    """
    ensure_data_dir()
    
    if end_year is None:
        end_year = datetime.now().strftime('%Y')
    
    # 标准化股票代码
    if '.' not in symbol:
        symbol = 'sh.' + symbol if symbol.startswith('6') else 'sz.' + symbol
    
    code = symbol.split('.')[1]
    filename = f"{code}_dividend.csv"
    filepath = DATA_DIR / filename
    
    # 检查本地缓存
    if filepath.exists() and not force:
        df = pd.read_csv(filepath)
        if len(df) > 0:
            print(f"使用本地缓存分红数据: {filename} ({len(df)} 条记录)")
            return df
    
    print(f"从 baostock 下载分红数据: {symbol}")
    
    bs.login()
    
    all_data = []
    for year in range(int(start_year), int(end_year) + 1):
        rs = bs.query_dividend_data(code=symbol, year=str(year), yearType="report")
        while rs.error_code == '0' and rs.next():
            all_data.append(rs.get_row_data())
    
    bs.logout()
    
    if not all_data:
        print(f"未找到分红数据: {symbol}")
        return pd.DataFrame()
    
    # 分红数据字段 - baostock 正确的字段顺序
    # Fields: code, dividPreNoticeDate, dividAgmPumDate, dividPlanAnnounceDate,
    #         dividPlanDate, dividRegistDate, dividOperateDate, dividPayDate,
    #         dividStockMarketDate, dividCashPsBeforeTax, dividCashPsAfterTax,
    #         dividStocksPs, dividCashStock, dividReserveToStockPs
    columns = ['code', 'dividPreNoticeDate', 'dividAgmPumDate', 'dividPlanAnnounceDate',
               'dividPlanDate', 'dividRegistDate', 'dividOperateDate', 'dividPayDate',
               'dividStockMarketDate', 'dividCashPsBeforeTax', 'dividCashPsAfterTax',
               'dividStocksPs', 'dividCashStock', 'dividReserveToStockPs']
    
    df = pd.DataFrame(all_data, columns=columns[:len(all_data[0])])
    
    # 解析分红金额
    # dividCashPsBeforeTax: 每股税前分红 (元)
    df['dividend_per_share'] = pd.to_numeric(df['dividCashPsBeforeTax'], errors='coerce').fillna(0)
    
    # 解析股权登记日
    # dividRegistDate: 股权登记日
    df['ex_date'] = pd.to_datetime(df['dividRegistDate'], format='%Y-%m-%d', errors='coerce')
    
    # 保存到本地
    df.to_csv(filepath, index=False)
    print(f"已保存分红数据: {filepath} ({len(df)} 条记录)")
    
    # 打印调试信息
    valid_div = df[df['dividend_per_share'] > 0]
    if len(valid_div) > 0:
        sample = valid_div.iloc[0]
        print(f"分红示例: {sample['dividCashStock']} -> {sample['dividend_per_share']}元/股, 登记日: {sample['ex_date']}")
    
    return df

def load_dividend_data(symbol, start_year='2020', end_year=None, auto_download=True):
    """
    加载分红数据
    
    Args:
        symbol: 股票代码
        start_year: 开始年份
        end_year: 结束年份
        auto_download: 是否自动下载
    
    Returns:
        DataFrame
    """
    ensure_data_dir()
    
    if end_year is None:
        end_year = datetime.now().strftime('%Y')
    
    if '.' not in symbol:
        symbol = 'sh.' + symbol if symbol.startswith('6') else 'sz.' + symbol
    
    code = symbol.split('.')[1]
    filename = f"{code}_dividend.csv"
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"加载本地分红数据: {filename} ({len(df)} 条记录)")
        return df
    
    if auto_download:
        return download_dividend_data(symbol, start_year, end_year)
    
    return pd.DataFrame()

def calculate_dividend_income(price_df, dividend_df, position_col='units'):
    """
    计算持仓期间的分红收入
    
    Args:
        price_df: 价格数据 DataFrame (需要有 date 索引和 close 列)
        dividend_df: 分红数据 DataFrame
        position_col: 持仓数量列名
    
    Returns:
        包含分红收入的 DataFrame
    """
    if dividend_df.empty or 'dividPreTax' not in dividend_df.columns:
        return price_df
    
    # 解析分红日期
    dividend_df = dividend_df.copy()
    dividend_df['dividRegistDate'] = pd.to_datetime(dividend_df['dividRegistDate'], errors='coerce')
    dividend_df['dividPreTax'] = pd.to_numeric(dividend_df['dividPreTax'], errors='coerce')
    
    # 有效分红记录
    valid_div = dividend_df[dividend_df['dividRegistDate'].notna() & dividend_df['dividPreTax'].notna()]
    
    if valid_div.empty:
        return price_df
    
    # 添加分红列
    price_df = price_df.copy()
    price_df['dividend'] = 0.0
    price_df['dividend_income'] = 0.0
    
    for _, div_row in valid_div.iterrows():
        regist_date = div_row['dividRegistDate']
        div_amount = div_row['dividPreTax']  # 每股分红金额
        
        # 找到登记日在价格数据中的位置
        if regist_date in price_df.index:
            price_df.loc[regist_date, 'dividend'] = div_amount
    
    return price_df

def list_local_data():
    """列出所有本地缓存的数据文件"""
    ensure_data_dir()
    files = list(DATA_DIR.glob('*.csv'))
    if not files:
        print("暂无本地数据")
        return []
    
    print(f"\n本地数据文件 ({len(files)} 个):")
    print("-" * 50)
    for f in sorted(files):
        df = pd.read_csv(f, index_col='date' if 'date' in pd.read_csv(f, nrows=1).columns else None, parse_dates=True)
        if df.index.name == 'date':
            print(f"  {f.name}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        else:
            print(f"  {f.name}: {len(df)} 条")
    print("-" * 50)
    return files

if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("数据管理模块")
    print("=" * 60)
    
    # 列出本地数据
    list_local_data()
    
    # 示例：下载或更新数据
    symbols = ['sh.601138', 'sh.512890', 'sh.159201']
    print(f"\n下载/更新股票数据: {symbols}")
    
    for symbol in symbols:
        df = load_stock_data(symbol, start_date='2020-01-01')
        if df is not None:
            print(f"\n{symbol} 数据预览:")
            print(df.tail())