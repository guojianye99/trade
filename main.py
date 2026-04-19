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
import rsi_strategy
import boll_strategy
import kdj_strategy

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


def run_rsi(symbol, capital):
    """运行 RSI 策略"""
    try:
        result = rsi_strategy.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"RSI 策略执行失败: {e}")
        return None


def run_boll(symbol, capital):
    """运行布林带策略"""
    try:
        result = boll_strategy.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"布林带策略执行失败: {e}")
        return None


def run_kdj(symbol, capital):
    """运行 KDJ 策略"""
    try:
        result = kdj_strategy.run(symbol=symbol, start_date=START_DATE, capital=capital)
        return result
    except Exception as e:
        print(f"KDJ 策略执行失败: {e}")
        return None


def print_summary(results, symbol, capital, user_shares=0, user_cost=0.0):
    """打印汇总结果
    
    Args:
        results: 各策略的结果
        symbol: 股票代码
        capital: 初始资金
        user_shares: 用户实际持仓股数
        user_cost: 用户持仓成本价
    """
    print("\n")
    print("=" * 70)
    print("策略汇总报告")
    print("=" * 70)
    print(f"股票代码: {symbol}")
    print(f"初始资金: {capital:,} 元")
    print(f"回测区间: {START_DATE} ~ {datetime.now().strftime('%Y-%m-%d')}")
    if user_shares > 0 and user_cost > 0:
        print(f"用户持仓: {user_shares} 股, 成本价: {user_cost:.2f} 元")
    elif user_shares > 0:
        print(f"用户持仓: {user_shares} 股 (未提供成本价)")
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
    
    if results['rsi']:
        r = results['rsi']
        print(f"{'RSI Strategy':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('RSI Strategy', r['return'], r['annual'], r))
    
    if results['boll']:
        r = results['boll']
        print(f"{'BOLL Strategy':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('BOLL Strategy', r['return'], r['annual'], r))
    
    if results['kdj']:
        r = results['kdj']
        print(f"{'KDJ Strategy':<20} {r['return']:>11.2f}% {r['annual']:>11.2f}% {r['trades']:>10} {r['dividends']:>11,.2f}")
        strategy_data.append(('KDJ Strategy', r['return'], r['annual'], r))
    
    print("-" * 70)
    
    # 找出最佳策略
    if strategy_data:
        # 按收益排序
        sorted_strategies = sorted(strategy_data, key=lambda x: x[1], reverse=True)
        best = sorted_strategies[0]
        print(f"最佳策略: {best[0]} (收益: {best[1]:.2f}%)")
        
        # 显示策略排名
        print("\n【策略表现排名】")
        for i, (name, ret, annual, _) in enumerate(sorted_strategies, 1):
            if ret > 0:
                print(f"  {i}. {name}: {ret:.2f}% (年化 {annual:.2f}%)")
            else:
                print(f"  {i}. {name}: {ret:.2f}% (亏损)")
    
    # 买卖建议汇总
    print("\n" + "=" * 70)
    print("【当前买卖建议汇总】")
    print("=" * 70)
    
    # 用户实际持仓状态
    has_user_position = user_shares > 0
    
    for strategy_name, total_return, annual_return, result in strategy_data:
        signal = result.get('current_signal', {})
        trend = result.get('trend', {})
        
        sig = signal.get('signal', '')
        confidence = signal.get('confidence', 0)
        current_price = signal.get('current_price', 0)
        
        # 计算盈亏情况
        profit_loss = 0.0
        profit_loss_pct = 0.0
        if has_user_position and user_cost > 0:
            profit_loss = (current_price - user_cost) * user_shares
            profit_loss_pct = (current_price - user_cost) / user_cost * 100
        
        print(f"\n>>> {strategy_name}")
        print(f"    历史收益: {total_return:.2f}% | 年化收益: {annual_return:.2f}%")
        print(f"    当前价格: {current_price:.2f}")
        print(f"    当前信号: {sig} | 信号强度: {confidence:.0f}%")
        print(f"    趋势判断: {trend.get('trend', 'N/A')} (强度: {trend.get('strength', 'N/A')})")
        
        # 显示持仓和盈亏信息
        if has_user_position:
            if user_cost > 0:
                profit_status = "盈利" if profit_loss >= 0 else "亏损"
                print(f"    您的持仓: {user_shares}股 | 成本价: {user_cost:.2f}元 | 市值: {user_shares * current_price:,.0f}元")
                print(f"    持仓盈亏: {profit_loss:+,.0f}元 ({profit_loss_pct:+.2f}%) - {profit_status}")
            else:
                print(f"    您的持仓: {user_shares}股 | 市值: {user_shares * current_price:,.0f}元 (未提供成本价)")
        else:
            print(f"    您的持仓: 空仓")
        
        # 根据策略类型、信号和用户实际持仓状态给出具体操作建议
        if sig == '买入':
            if has_user_position:
                print(f"    ★★★ 操作建议: 继续持有或加仓 ★★★")
                if user_cost > 0:
                    print(f"    当前已有持仓: {user_shares}股 (成本 {user_cost:.2f}元)")
                    if profit_loss >= 0:
                        print(f"    当前盈利 {profit_loss_pct:+.2f}%，策略发出买入信号，可继续持有或考虑加仓")
                    else:
                        print(f"    当前亏损 {profit_loss_pct:.2f}%，策略发出买入信号，建议持有等待反弹")
                else:
                    print(f"    当前已有持仓: {user_shares}股")
                    print(f"    建议: 策略发出买入信号，可继续持有或考虑加仓")
            else:
                print(f"    ★★★ 操作建议: 买入 ★★★")
                print(f"    参考买入价: {current_price:.2f} 元")
                # 根据不同策略给出止损止盈建议
                if 'upper' in signal and 'lower' in signal:
                    print(f"    止损参考: {signal['lower']:.2f} 元 (下轨)")
                elif 'ma_long' in signal:
                    print(f"    止损参考: {signal['ma_long']:.2f} 元 (长期均线)")
                print(f"    建议仓位: 可分批建仓，首次 30-50%")
            
        elif sig == '卖出':
            if has_user_position:
                print(f"    ★★★ 操作建议: 卖出 ★★★")
                print(f"    参考卖出价: {current_price:.2f} 元")
                print(f"    预计回笼资金: 约 {user_shares * current_price:,.0f} 元")
                if user_cost > 0:
                    print(f"    预计盈亏: {profit_loss:+,.0f} 元 ({profit_loss_pct:+.2f}%)")
                    if profit_loss > 0:
                        print(f"    当前盈利，建议卖出锁定收益")
                    elif profit_loss < 0:
                        print(f"    当前亏损，建议止损离场，控制风险")
                print(f"    建议操作: 建议全部卖出或减仓 70%")
            else:
                print(f"    ★★★ 操作建议: 继续观望 ★★★")
                print(f"    当前无持仓，卖出信号不适用")
                print(f"    建议: 继续观望，等待买入机会")
            
        elif '偏买' in sig:
            if has_user_position:
                print(f"    ★★ 操作建议: 继续持有 ★★")
                if user_cost > 0:
                    print(f"    当前已有持仓: {user_shares}股 (成本 {user_cost:.2f}元)")
                    if profit_loss > 0:
                        print(f"    当前盈利 {profit_loss_pct:+.2f}%，建议继续持有，享受上涨趋势")
                    else:
                        print(f"    当前亏损 {profit_loss_pct:.2f}%，建议持有等待反弹")
                else:
                    print(f"    当前已有持仓: {user_shares}股")
                    print(f"    建议: 继续持有，享受上涨趋势")
            else:
                print(f"    ★★ 操作建议: 关注买入机会 ★★")
                # 显示关键价位
                if 'upper' in signal:
                    print(f"    突破买入价: {signal['upper']:.2f} 元")
                    print(f"    回调支撑价: {signal.get('lower', current_price * 0.95):.2f} 元")
                elif 'ma_short' in signal and 'ma_long' in signal:
                    print(f"    参考买入价: {signal['ma_long']:.2f} 元附近 (长期均线支撑)")
                print(f"    建议仓位: 小仓位试探 20-30%，突破后加仓")
                print(f"    止损建议: 若买入后价格下跌 5-8%，考虑止损离场")
            
        elif '偏卖' in sig:
            if has_user_position:
                print(f"    ★★ 操作建议: 注意风险，考虑减仓 ★★")
                if user_cost > 0:
                    print(f"    当前持仓: {user_shares}股 (成本 {user_cost:.2f}元)")
                    if profit_loss > 0:
                        print(f"    当前盈利 {profit_loss_pct:+.2f}%，建议减仓 30-50% 锁定部分收益")
                    else:
                        print(f"    当前亏损 {profit_loss_pct:.2f}%，建议减仓控制风险，等待更好时机")
                else:
                    print(f"    当前持仓: {user_shares}股")
                
                if 'rsi' in signal:
                    print(f"    RSI 值: {signal['rsi']:.1f} (超买区)")
                print(f"    止盈参考: 当前价位或上涨 3-5%")
            else:
                print(f"    ★★ 操作建议: 继续观望 ★★")
                print(f"    当前无持仓，偏卖信号不适用")
                print(f"    建议: 继续观望，等待更好的买入时机")
            
        else:
            if has_user_position:
                print(f"    ★ 操作建议: 继续持有 ★")
                if user_cost > 0:
                    print(f"    当前已有持仓: {user_shares}股 (成本 {user_cost:.2f}元)")
                    if profit_loss > 0:
                        print(f"    当前盈利 {profit_loss_pct:+.2f}%，建议继续持有，设置止盈保护利润")
                        # 根据盈利比例给出不同的止盈建议
                        if profit_loss_pct >= 15:
                            # 盈利超过15%，建议减仓锁定部分收益
                            print(f"    止盈建议: 已有较大盈利，建议减仓 30-50% 锁定收益")
                            print(f"    或设置移动止盈: 当价格回落至盈利 10% 时卖出")
                        elif profit_loss_pct >= 10:
                            # 盈利10-15%，建议设置止盈保护
                            print(f"    止盈建议: 可设置移动止盈，当价格回落至盈利 5-8% 时减仓")
                        else:
                            # 盈利10%以下，建议设置保本止损
                            print(f"    止盈建议: 可设置保本止损，确保不亏损")
                            print(f"    具体价位: 成本价 {user_cost:.2f} 元附近")
                    else:
                        print(f"    当前亏损 {profit_loss_pct:.2f}%，建议持有，设置止损控制风险")
                        # 根据亏损比例给出不同的止损建议
                        loss_pct = abs(profit_loss_pct)
                        if loss_pct >= 10:
                            # 亏损超过10%，建议考虑止损
                            print(f"    止损建议: 亏损较大，建议考虑止损离场或减仓控制风险")
                        elif loss_pct >= 5:
                            # 亏损5-10%，建议设置止损线
                            stop_loss_price = user_cost * 0.92  # 成本价下方8%
                            print(f"    止损建议: 设置止损线在成本价下方 8-10%")
                            print(f"    具体价位: {stop_loss_price:.2f} 元附近")
                        else:
                            # 亏损5%以下，可以等待反弹
                            print(f"    止损建议: 亏损较小，可设置止损线在成本价下方 10%")
                            stop_loss_price = user_cost * 0.90
                            print(f"    具体价位: {stop_loss_price:.2f} 元附近")
                else:
                    print(f"    当前已有持仓: {user_shares}股")
                    print(f"    建议: 继续持有，设置止损保护")
            else:
                print(f"    ★ 操作建议: 观望 ★")
                print(f"    等待明确信号再操作")
                print(f"    建议: 暂不操作，等待机会")
    
    # 综合建议
    print("\n" + "-" * 70)
    print("【综合建议】")
    
    buy_count = sum(1 for _, _, _, r in strategy_data if r.get('current_signal', {}).get('signal') == '买入')
    sell_count = sum(1 for _, _, _, r in strategy_data if r.get('current_signal', {}).get('signal') == '卖出')
    bias_buy = sum(1 for _, _, _, r in strategy_data if '偏买' in r.get('current_signal', {}).get('signal', ''))
    bias_sell = sum(1 for _, _, _, r in strategy_data if '偏卖' in r.get('current_signal', {}).get('signal', ''))
    
    # 获取最佳策略的信号
    sorted_strategies = sorted(strategy_data, key=lambda x: x[1], reverse=True)
    best_strategy = sorted_strategies[0]
    best_signal = best_strategy[3].get('current_signal', {}).get('signal', '')
    
    print(f"\n该股票最适合策略: {best_strategy[0]}")
    print(f"  - 历史收益: {best_strategy[1]:.2f}%")
    print(f"  - 年化收益: {best_strategy[2]:.2f}%")
    print(f"  - 当前信号: {best_signal}")
    
    print(f"\n信号统计:")
    print(f"  - 买入信号: {buy_count} 个")
    print(f"  - 偏买信号: {bias_buy} 个")
    print(f"  - 偏卖信号: {bias_sell} 个")
    print(f"  - 卖出信号: {sell_count} 个")
    
    # 基于最佳策略给出建议
    print(f"\n操作建议:")
    if best_signal == '买入':
        print(f"  ★★★ {best_strategy[0]} 发出买入信号，建议买入 ★★★")
    elif best_signal == '卖出':
        print(f"  ★★★ {best_strategy[0]} 发出卖出信号，建议卖出 ★★★")
    elif '偏买' in best_signal:
        print(f"  ★★ {best_strategy[0]} 显示偏多，关注买入机会 ★★")
    elif '偏卖' in best_signal:
        print(f"  ★★ {best_strategy[0]} 显示偏空，注意风险 ★★")
    else:
        print(f"  ★ {best_strategy[0]} 建议: 观望 ★")
    
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
    
    # 用户实际持仓股数(可选参数)
    user_shares = 0
    if len(sys.argv) > 3:
        try:
            user_shares = int(sys.argv[3])
        except:
            pass
    
    # 用户持仓成本价(可选参数)
    user_cost = 0.0
    if len(sys.argv) > 4:
        try:
            user_cost = float(sys.argv[4])
        except:
            pass
    
    print("=" * 70)
    print("量化策略分析工具")
    print("=" * 70)
    print(f"目标股票: {symbol}")
    print(f"初始资金: {capital:,} 元")
    if user_shares > 0 and user_cost > 0:
        print(f"用户持仓: {user_shares} 股, 成本价: {user_cost:.2f} 元")
    elif user_shares > 0:
        print(f"用户持仓: {user_shares} 股")
    
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
        'rsi': None,
        'boll': None,
        'kdj': None,
        'rebalance': None
    }
    
    # Dual Thrust
    print(f"\n[1/6] Dual Thrust 策略...")
    results['dual_thrust'] = run_dual_thrust(symbol, capital)
    
    # MA Crossover
    print(f"\n[2/6] 均线交叉策略...")
    results['ma_crossover'] = run_ma_crossover(symbol, capital)
    
    # MACD
    print(f"\n[3/6] MACD 策略...")
    results['macd'] = run_macd(symbol, capital)
    
    # RSI
    print(f"\n[4/6] RSI 策略...")
    results['rsi'] = run_rsi(symbol, capital)
    
    # BOLL
    print(f"\n[5/6] 布林带策略...")
    results['boll'] = run_boll(symbol, capital)
    
    # KDJ
    print(f"\n[6/6] KDJ 策略...")
    results['kdj'] = run_kdj(symbol, capital)
    
    # 打印汇总
    print_summary(results, symbol, capital, user_shares, user_cost)


if __name__ == "__main__":
    main()