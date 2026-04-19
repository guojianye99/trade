"""
数据下载脚本 - 批量下载股票数据并保存到本地
使用方法: python download_data.py
"""
import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import download_stock_data, download_multiple, list_local_data, DATA_DIR

# 默认下载的股票列表
STOCKS = {
    '601138': 'sh.601138',  # 工商银行
    '512890': 'sh.512890',  # 科创50ETF
    '159201': 'sz.159201',  # 创业板ETF
}

# 默认日期范围
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

def download_all(symbols=None, start_date=None, end_date=None, force=False):
    """下载所有股票数据
    
    Args:
        symbols: 股票代码字典 {name: symbol}
        start_date: 开始日期
        end_date: 结束日期
        force: 是否强制重新下载
    """
    symbols = symbols or STOCKS
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    
    print("=" * 60)
    print("股票数据下载工具")
    print("=" * 60)
    print(f"数据存储目录: {DATA_DIR}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"股票数量: {len(symbols)}")
    print("=" * 60)
    
    # 显示将要下载的股票
    print("\n待下载股票列表:")
    for name, symbol in symbols.items():
        print(f"  {name}: {symbol}")
    
    print("\n开始下载...\n")
    
    # 批量下载
    symbol_list = list(symbols.values())
    results = download_multiple(symbol_list, start_date, end_date, force=force)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"成功: {len(results)} 只")
    print(f"失败: {len(symbol_list) - len(results)} 只")
    
    # 列出本地数据
    print("\n当前本地数据:")
    list_local_data()
    
    return results

def update_data():
    """更新本地数据 (只更新，不强制重新下载)"""
    print("=" * 60)
    print("更新本地数据...")
    print("=" * 60)
    return download_all(force=False)

def force_refresh():
    """强制刷新所有数据"""
    print("=" * 60)
    print("强制刷新所有数据...")
    print("=" * 60)
    return download_all(force=True)

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--update':
            update_data()
        elif sys.argv[1] == '--force':
            force_refresh()
        elif sys.argv[1] == '--list':
            list_local_data()
        else:
            print("用法:")
            print("  python download_data.py          # 下载缺失的数据")
            print("  python download_data.py --update # 更新本地数据")
            print("  python download_data.py --force  # 强制重新下载")
            print("  python download_data.py --list   # 列出本地数据")
    else:
        # 默认下载
        download_all()