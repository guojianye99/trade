#!/usr/bin/env python3
"""
BOLL 布林带策略
- 价格突破上轨: 超买，可能回调
- 价格跌破下轨: 超卖，可能反弹
- 价格在轨道内: 正常波动
"""

import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 布林带参数
BOLL_PERIOD = 20
BOLL_STD = 2.0  # 标准差倍数


def calculate_boll(df, period=20, std_dev=2.0):
    """计算布林带指标"""
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    
    return upper, middle, lower


def boll_strategy(df, period=20, std_dev=2.0):
    """布林带策略
    
    买入信号: 价格从下轨下方上穿下轨
    卖出信号: 价格从上轨上方下穿上轨
    """
    df = df.copy()
    df['boll_upper'], df['boll_middle'], df['boll_lower'] = calculate_boll(df, period, std_dev)
    
    df['prev_close'] = df['close'].shift(1)
    df['prev_upper'] = df['boll_upper'].shift(1)
    df['prev_lower'] = df['boll_lower'].shift(1)
    
    df['position'] = 'hold'
    # 价格从下轨下方上穿下轨，买入
    df.loc[(df['close'] > df['boll_lower']) & (df['prev_close'] <= df['prev_lower']), 'position'] = 'buy'
    # 价格从上轨上方下穿上轨，卖出
    df.loc[(df['close'] < df['boll_upper']) & (df['prev_close'] >= df['prev_upper']), 'position'] = 'sell'
    
    records = []
    for i in range(period, len(df)):
        if df['position'].iloc[i] != 'hold':
            records.append({
                'date': df.index[i],
                'close': df['close'].iloc[i],
                'upper': df['boll_upper'].iloc[i],
                'middle': df['boll_middle'].iloc[i],
                'lower': df['boll_lower'].iloc[i],
                'position': df['position'].iloc[i]
            })
    
    # 如果没有信号，返回带正确列的空 DataFrame
    if records:
        return pd.DataFrame(records).set_index('date'), df
    else:
        return pd.DataFrame(columns=['close', 'upper', 'middle', 'lower', 'position']), df


def get_current_signal(df, period=20, std_dev=2.0):
    """获取当前买卖信号"""
    if len(df) < period:
        return {'signal': '数据不足', 'confidence': 0}
    
    upper, middle, lower = calculate_boll(df, period, std_dev)
    current_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    current_middle = middle.iloc[-1]
    prev_upper = upper.iloc[-2]
    prev_lower = lower.iloc[-2]
    
    # 计算带宽
    bandwidth = (current_upper - current_lower) / current_middle * 100
    
    # 计算价格在布林带中的位置 (0-100%)
    boll_position = (current_close - current_lower) / (current_upper - current_lower) * 100 if current_upper != current_lower else 50
    
    # 计算趋势
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    is_uptrend = ma5 > ma10
    
    # 判断信号
    if current_close > current_upper and prev_close <= prev_upper:
        signal = '买入'  # 突破上轨
        confidence = min(100, 75)
    elif current_close < current_lower and prev_close >= prev_lower:
        signal = '卖出'  # 跌破下轨
        confidence = min(100, 75)
    elif current_close > current_lower and prev_close <= prev_lower:
        signal = '买入'  # 从下轨反弹
        confidence = min(100, 70)
    elif current_close < current_upper and prev_close >= prev_upper:
        signal = '卖出'  # 从上轨回落
        confidence = min(100, 70)
    else:
        # 根据价格在布林带中的位置判断
        if boll_position > 70 or (boll_position > 50 and is_uptrend):
            signal = '观望(偏买)'
            confidence = min(100, 55 + boll_position * 0.3)
        elif boll_position < 30 or (boll_position < 50 and not is_uptrend):
            signal = '观望(偏卖)'
            confidence = min(100, 55 + (100 - boll_position) * 0.3)
        else:
            signal = '观望'
            confidence = 50
    
    return {
        'signal': signal,
        'confidence': confidence,
        'upper': current_upper,
        'middle': current_middle,
        'lower': current_lower,
        'current_price': current_close,
        'bandwidth': bandwidth,
        'boll_position': boll_position
    }


def analyze_trend(df):
    """分析趋势"""
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    current = df['close'].iloc[-1]
    
    if ma5 > ma10 > ma20:
        trend = '上升趋势'
        strength = '强'
    elif ma5 > ma10:
        trend = '上升趋势'
        strength = '中'
    elif ma5 < ma10 < ma20:
        trend = '下降趋势'
        strength = '强'
    elif ma5 < ma10:
        trend = '下降趋势'
        strength = '中'
    else:
        trend = '横盘整理'
        strength = '弱'
    
    change_5d = (current / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    change_20d = (current / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
    
    return {
        'trend': trend,
        'strength': strength,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'change_5d': change_5d,
        'change_20d': change_20d
    }


def run(symbol=None, start_date=None, end_date=None, capital=None, 
        period=None, std_dev=None):
    """运行布林带策略回测"""
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    period = period or BOLL_PERIOD
    std_dev = std_dev or BOLL_STD
    
    print("=" * 60)
    print("BOLL Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"BOLL Period: {period}")
    print(f"Std Dev: {std_dev}")
    print(f"Initial: {capital}")
    print("=" * 60)
    
    df = load_stock_data(symbol, start_date, end_date)
    if df is None or len(df) == 0:
        print("错误: 无法获取数据")
        return None
    
    start_year = start_date[:4]
    end_year = end_date[:4]
    dividend_df = load_dividend_data(symbol, start_year, end_year)
    
    print(f"\nData loaded: {len(df)} records")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    if not dividend_df.empty:
        print(f"Dividend records: {len(dividend_df)}")
    
    signals, df_full = boll_strategy(df, period, std_dev)
    
    cash = capital
    position = 0
    total_dividend = 0
    trades = []
    
    if not dividend_df.empty and 'ex_date' in dividend_df.columns:
        dividend_df['ex_date'] = pd.to_datetime(dividend_df['ex_date'])
        div_dict = {}
        for _, row in dividend_df.iterrows():
            if pd.notna(row['ex_date']) and row.get('dividend_per_share', 0) > 0:
                div_dict[row['ex_date']] = row['dividend_per_share']
    else:
        div_dict = {}
    
    signal_dates = set(signals.index)
    current_position = 0
    
    for date in df.index:
        if date in div_dict and current_position > 0:
            div_per_share = div_dict[date]
            div_income = current_position * div_per_share
            cash += div_income
            total_dividend += div_income
            trades.append(('DIVIDEND', date, div_income, current_position))
        
        if date in signal_dates:
            signal_row = signals.loc[date]
            if signal_row['position'] == 'buy' and current_position == 0:
                units = cash / signal_row['close']
                current_position = units
                cash = 0
                trades.append(('BUY', date, signal_row['close'], current_position))
            elif signal_row['position'] == 'sell' and current_position > 0:
                value = current_position * signal_row['close']
                cash = value
                trades.append(('SELL', date, signal_row['close'], current_position))
                current_position = 0
    
    if current_position > 0:
        final_value = current_position * df['close'].iloc[-1]
    else:
        final_value = cash
    
    total_value = final_value + cash
    total_return = (total_value - capital) / capital * 100
    
    print(f"\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"Total Trades: {len([t for t in trades if t[0] in ['BUY', 'SELL']])}")
    print(f"Total Dividends: {total_dividend:,.2f}")
    print(f"Final Value: {total_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    first_date = signals.index[0]
    last_date = signals.index[-1]
    years = (last_date - first_date).days / 365
    annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    print(f"Annual Return: {annual_return:.2f}%")
    
    buy_hold = (signals['close'].iloc[-1] - signals['close'].iloc[0]) / signals['close'].iloc[0] * 100
    print(f"Buy&Hold Return: {buy_hold:.2f}%")
    print(f"Strategy vs B&H: {total_return - buy_hold:.2f}%")
    
    # 当前买卖建议
    print(f"\n" + "=" * 60)
    print("当前买卖建议")
    print("=" * 60)
    
    current_signal = get_current_signal(df, period, std_dev)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\n布林带:")
    print(f"  上轨: {current_signal['upper']:.2f}")
    print(f"  中轨: {current_signal['middle']:.2f}")
    print(f"  下轨: {current_signal['lower']:.2f}")
    print(f"  带宽: {current_signal['bandwidth']:.2f}%")
    print(f"  位置: {current_signal['boll_position']:.0f}%")
    
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    价格突破布林带")
        print(f"    参考买入价: {current_price:.2f} 附近")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    价格跌破布林带")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    价格在布林带上半部分 ({current_signal['boll_position']:.0f}%)")
        print()
        print("    - 如果已有持仓: 继续持有")
        print("    - 如果没有持仓: 关注突破机会")
    elif '偏卖' in signal:
        print("\n>>> 建议操作: 注意风险")
        print(f"    价格在布林带下半部分 ({current_signal['boll_position']:.0f}%)")
        print()
        print("    - 如果已有持仓: 注意风险控制")
        print("    - 如果没有持仓: 暂时观望")
    else:
        print("\n>>> 建议操作: 暂时观望，不操作")
        print(f"    价格在布林带中间区域")
        print()
        print("    - 如果已有持仓: 继续持有")
        print("    - 如果没有持仓: 等待明确信号")
    
    print("\n" + "-" * 40)
    print("Trade History:")
    for t in trades[:15]:
        if t[0] == 'DIVIDEND':
            print(f"  {t[0]} {t[1].strftime('%Y-%m-%d')} +{t[2]:.2f} ({t[3]:.0f}股)")
        else:
            print(f"  {t[0]} {t[1].strftime('%Y-%m-%d')} @ {t[2]:.2f}")
    if len(trades) > 15:
        print(f"  ... ({len(trades)-15} more)")
    
    return {
        'trades': len([t for t in trades if t[0] in ['BUY', 'SELL']]),
        'dividends': total_dividend,
        'final': total_value,
        'return': total_return,
        'annual': annual_return,
        'current_signal': current_signal,
        'trend': trend_info,
        'current_position': current_position,
        'current_shares': current_position
    }


if __name__ == "__main__":
    run()