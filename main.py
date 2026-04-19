#!/usr/bin/env python3
"""
量化策略统一入口
- 自动检测并更新本地数据
- 运行所有策略并汇总结果
- 输出交易收益和买卖建议
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from data_manager import load_stock_data, load_dividend_data, download_stock_data, download_dividend_data

# 导入各策略模块
import dual_thrust
import ma_crossover
import macd_strategy
import rebalance_backtest

# 默认配置
DEFAULT_SYMBOL = 'sh.601138'
DEFAULT_CAPITAL = 100000
START_DATE = '2020-01-01'

# 本地数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_local_data_info(symbol):
    """获取本地数据信息"""
    code = symbol.split('.')[1]
    csv_file = os.path.join(DATA_DIR, f'{code}.csv')
    
    if not os.path.exists(csv_file):
        return None
    
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return None
        
        # 获取最后日期
        last_date = pd.to_datetime(df['date'].iloc[-1])
        return {
            'file': csv_file,
            'last_date': last_date,
            'records': len(df)
        }
    except Exception as e:
        print(f"读取本地数据失败: {e}")
        return None


def is_data_fresh(symbol):
    """检查数据是否包含最近一个交易日"""
    info = get_local_data_info(symbol)
    if info is None:
        return False
    
    last_date = info['last_date']
    today = datetime.now()
    
    # 判断最后日期是否是最近一个交易日
    # 如果今天是周末，周五的数据也算新鲜
    if today.weekday() == 0:  # 周一
        # 需要上周五的数据
        expected_date = today - timedelta(days=3)
    elif today.weekday() == 6:  # 周日
        # 需要上周五的数据
        expected_date = today - timedelta(days=2)
    elif today.weekday() == 5:  # 周六
        # 需要昨天的数据（周五）
        expected_date = today - timedelta(days=1)
    else:
        # 工作日，需要昨天的数据
        expected_date = today - timedelta(days=1)
    
    # 只比较日期部分
    return last_date.date() >= expected_date.date()


def update_data(symbol):
    """更新股票数据"""
    print(f"\n正在更新 {symbol} 数据...")
    
    code = symbol.split('.')[1]
    
    # 下载价格数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        df = download_stock_data(symbol, START_DATE, end_date)
        if df is not None and len(df) > 0:
            # 保存到本地
            csv_file = os.path.join(DATA_DIR, f'{code}.csv')
            df.to_csv(csv_file)
            print(f"  价格数据已更新: {len(df)} 条记录")
        else:
            print(f"  价格数据下载失败")
            return False
    except Exception as e:
        print(f"  价格数据更新失败: {e}")
        return False
    
    # 下载分红数据
    try:
        start_year = START_DATE[:4]
        end_year = end_date[:4]
        div_df = download_dividend_data(symbol, start_year, end_year)
        if div_df is not None and len(div_df) > 0:
            div_file = os.path.join(DATA_DIR, f'{code}_dividend.csv')
            div_df.to_csv(div_file)
            print(f"  分红数据已更新: {len(div_df)} 条记录")
    except Exception as e:
        print(f"  分红数据更新失败: {e}")
    
    return True


def run_dual_thrust(symbol, capital):
    """运行 Dual Thrust 策略"""
    try:
        result = dual_thrust.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"Dual Thrust 策略执行失败: {e}")
        return None


def run_ma_crossover(symbol, capital):
    """运行均线交叉策略"""
    try:
        result = ma_crossover.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"均线交叉策略执行失败: {e}")
        return None


def run_macd(symbol, capital):
    """运行 MACD 策略"""
    try:
        result = macd_strategy.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"MACD 策略执行失败: {e}")
        return None


def run_rebalance(symbols, capital):
    """运行再平衡策略"""
    try:
        result = rebalance_backtest.run(symbols=symbols, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"再平衡策略执行失败: {e}")
        return None


def print_summary(results, symbol, capital):
    """打印汇总结果"""
    print("\n")
    print("=" * 70)
    print("策略汇总报告")
    print("=" * 70)
    print(f"股票代码: {symbol}")
    print(f"初始资金: {capital:,} 元")
    print(f"回测区间: {START_DATE} ~ {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    # 收益对比表
    print("\n【策略收益对比】")
    print("-" * 70)
    print(f"{'策略名称':<20} {'总收益':>12} {'年化收益':>12} {'交易次数':>10} {'分红收入':>12}")
    print("-" * 70)
    
    strategy_data = []
    
    if results['dual_thrust']:
        r = results['dual_thrust']
        print(f"{'Dual Thrust':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('Dual Thrust', r['return'], r['annual'], r))
    
    if results['ma_crossover']:
        r = results['ma_crossover']
        print(f"{'MA Crossover':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('MA Crossover', r['return'], r['annual'], r))
    
    if results['macd']:
        r = results['macd']
        print(f"{'MACD Strategy':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('MACD Strategy', r['return'], r['annual'], r))
    
    print("-" * 70)
    
    # 找出最佳策略
    if strategy_data:
        best = max(strategy_data, key=lambda x: x[1])
        print(f"最佳策略: {best[0]} (收益: {best[1]:.2f}%)")
    
    # 买卖建议汇总
    print("\n" + "=" * 70)
    print("【当前买卖建议汇总】")
    print("=" * 70)
    
    for strategy_name, _, _, result in strategy_data:
        signal = result.get('current_signal', {})
        trend = result.get('trend', {})
        
        sig = signal.get('signal', '')
        confidence = signal.get('confidence', 0)
        
        print(f"\n>>> {strategy_name}")
        print(f"    当前信号: {sig}")
        print(f"    信号强度: {confidence:.0f}%")
        print(f"    趋势判断: {trend.get('trend', 'N/A')} (强度: {trend.get('strength', 'N/A')})")
        
        # 根据信号给出具体建议
        if sig == '买入':
            print(f"    ★★★ 建议: 买入 ★★★")
        elif sig == '卖出':
            print(f"    ★★★ 建议: 卖出 ★★★")
        elif '偏买' in sig:
            print(f"    ★★ 建议: 关注买入机会 (多头趋势) ★★")
        elif '偏卖' in sig:
            print(f"    ★★ 建议: 注意风险 (空头趋势) ★★")
        else:
            print(f"    ★ 建议: 观望 (有仓持有，无仓等待) ★")
    
    # 综合建议
    print("\n" + "-" * 70)
    print("【综合建议】")
    
    buy_count = sum(1 for _, _, _, r in strategy_data if r.get('current_signal', {}).get('signal') == '买入')
    sell_count = sum(1 for _, _, _, r in strategy_data if r.get('current_signal', {}).get('signal') == '卖出')
    bias_buy = sum(1 for _, _, _, r in strategy_data if '偏买' in r.get('current_signal', {}).get('signal', ''))
    bias_sell = sum(1 for _, _, _, r in strategy_data if '偏卖' in r.get('current_signal', {}).get('signal', ''))
    
    if buy_count >= 2:
        print("  多个策略发出买入信号，建议: 积极买入")
    elif sell_count >= 2:
        print("  多个策略发出卖出信号，建议: 及时卖出")
    elif bias_buy >= 2:
        print("  多个策略显示偏多信号，建议: 关注买入机会")
    elif bias_sell >= 2:
        print("  多个策略显示偏空信号，建议: 注意风险")
    else:
        print("  各策略信号分化，建议: 保持观望，等待明确信号")
    
    print("=" * 70)


def main():
    """主函数"""
    # 获取股票代码
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        # 处理输入格式
        if '.' not in symbol:
            # 纯数字代码，添加前缀
            if symbol.startswith('6'):
                symbol = f'sh.{symbol}'
            else:
                symbol = f'sz.{symbol}'
    else:
        symbol = DEFAULT_SYMBOL
    
    capital = DEFAULT_CAPITAL
    if len(sys.argv) > 2:
        try:
            capital = int(sys.argv[2])
        except:
            pass
    
    print("=" * 70)
    print("量化策略分析工具")
    print("=" * 70)
    print(f"目标股票: {symbol}")
    print(f"初始资金: {capital:,} 元")
    
    # 检查数据
    print(f"\n检查本地数据...")
    local_info = get_local_data_info(symbol)
    
    if local_info:
        print(f"  本地数据: {local_info['records']} 条记录")
        print(f"  最后日期: {local_info['last_date'].strftime('%Y-%m-%d')}")
        
        if is_data_fresh(symbol):
            print(f"  数据状态: ✓ 最新")
        else:
            print(f"  数据状态: ✗ 需要更新")
            if not update_data(symbol):
                print("数据更新失败，使用本地数据继续分析")
    else:
        print(f"  本地数据: 无")
        print(f"  数据状态: 需要下载")
        if not update_data(symbol):
            print("数据下载失败，无法继续分析")
            return
    
    # 运行所有策略
    print(f"\n运行策略分析...")
    
    results = {
        'dual_thrust': None,
        'ma_crossover': None,
        'macd': None,
        'rebalance': None
    }
    
    # Dual Thrust
    print(f"\n[1/3] Dual Thrust 策略...")
    results['dual_thrust'] = run_dual_thrust(symbol, capital)
    
    # MA Crossover
    print(f"\n[2/3] 均线交叉策略...")
    results['ma_crossover'] = run_ma_crossover(symbol, capital)
    
    # MACD
    print(f"\n[3/3] MACD 策略...")
    results['macd'] = run_macd(symbol, capital)
    
    # 打印汇总
    print_summary(results, symbol, capital)


if __name__ == "__main__":
    main()