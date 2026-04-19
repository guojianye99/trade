#!/usr/bin/env python3
"""
KDJ 随机指标策略
- K线: 快速指标
- D线: 慢速指标
- J线: 超前指标
- K > D: 多头趋势
- K < D: 空头趋势
- KDJ > 80: 超买
- KDJ < 20: 超卖
"""

import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# KDJ 参数
KDJ_N = 9      # RSV 计算周期
KDJ_M1 = 3     # K 值平滑周期
KDJ_M2 = 3     # D 值平滑周期


def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标"""
    # 计算 RSV
    low_n = df['low'].rolling(window=n, min_periods=1).min()
    high_n = df['high'].rolling(window=n, min_periods=1).max()
    
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    
    # 计算 K, D, J
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def kdj_strategy(df, n=9, m1=3, m2=3):
    """KDJ 策略
    
    买入信号: 
    - K线上穿D线且K < 20 (超卖区金叉)
    - J线从下往上穿过0
    
    卖出信号:
    - K线下穿D线且K > 80 (超买区死叉)
    - J线从上往下穿过100
    """
    df = df.copy()
    df['k'], df['d'], df['j'] = calculate_kdj(df, n, m1, m2)
    
    df['prev_k'] = df['k'].shift(1)
    df['prev_d'] = df['d'].shift(1)
    df['prev_j'] = df['j'].shift(1)
    
    df['position'] = 'hold'
    # 超卖区金叉，买入
    df.loc[(df['k'] > df['d']) & (df['prev_k'] <= df['prev_d']) & (df['k'] < 30), 'position'] = 'buy'
    # 超买区死叉，卖出
    df.loc[(df['k'] < df['d']) & (df['prev_k'] >= df['prev_d']) & (df['k'] > 70), 'position'] = 'sell'
    
    records = []
    for i in range(n, len(df)):
        if df['position'].iloc[i] != 'hold':
            records.append({
                'date': df.index[i],
                'close': df['close'].iloc[i],
                'k': df['k'].iloc[i],
                'd': df['d'].iloc[i],
                'j': df['j'].iloc[i],
                'position': df['position'].iloc[i]
            })
    
    # 如果没有信号，返回带正确列的空 DataFrame
    if records:
        return pd.DataFrame(records).set_index('date'), df
    else:
        return pd.DataFrame(columns=['close', 'k', 'd', 'j', 'position']), df


def get_current_signal(df, n=9, m1=3, m2=3):
    """获取当前买卖信号"""
    if len(df) < n:
        return {'signal': '数据不足', 'confidence': 0}
    
    k, d, j = calculate_kdj(df, n, m1, m2)
    
    current_k = k.iloc[-1]
    current_d = d.iloc[-1]
    current_j = j.iloc[-1]
    prev_k = k.iloc[-2]
    prev_d = d.iloc[-2]
    current_price = df['close'].iloc[-1]
    
    # 计算趋势
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    is_uptrend = ma5 > ma10
    
    # 判断信号
    if current_k > current_d and prev_k <= prev_d and current_k < 30:
        signal = '买入'  # 超卖区金叉
        confidence = min(100, 75 + (30 - current_k))
    elif current_k < current_d and prev_k >= prev_d and current_k > 70:
        signal = '卖出'  # 超买区死叉
        confidence = min(100, 75 + (current_k - 70))
    else:
        # 根据KDJ位置判断倾向
        if current_k > current_d:
            if current_k < 40:
                signal = '观望(偏买)'
                confidence = min(100, 60 + (40 - current_k))
            else:
                signal = '观望'
                confidence = 55
        else:
            if current_k > 60:
                signal = '观望(偏卖)'
                confidence = min(100, 60 + (current_k - 60))
            else:
                signal = '观望'
                confidence = 55
    
    return {
        'signal': signal,
        'confidence': confidence,
        'k': current_k,
        'd': current_d,
        'j': current_j,
        'current_price': current_price
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
        n=None, m1=None, m2=None):
    """运行 KDJ 策略回测"""
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    n = n or KDJ_N
    m1 = m1 or KDJ_M1
    m2 = m2 or KDJ_M2
    
    print("=" * 60)
    print("KDJ Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"KDJ N: {n}, M1: {m1}, M2: {m2}")
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
    
    signals, df_full = kdj_strategy(df, n, m1, m2)
    
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
    
    if len(signals) > 0:
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
    
    current_signal = get_current_signal(df, n, m1, m2)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\nKDJ指标:")
    print(f"  K: {current_signal['k']:.2f}")
    print(f"  D: {current_signal['d']:.2f}")
    print(f"  J: {current_signal['j']:.2f}")
    
    # 判断超买超卖
    k_val = current_signal['k']
    if k_val > 80:
        print(f"  状态: 超买区域 (K > 80)")
    elif k_val < 20:
        print(f"  状态: 超卖区域 (K < 20)")
    else:
        print(f"  状态: 正常区域")
    
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    超卖区金叉 (K={k_val:.2f} 上穿 D)")
        print(f"    参考买入价: {current_price:.2f} 附近")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    超买区死叉 (K={k_val:.2f} 下穿 D)")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    K > D，K在较低位置 ({k_val:.2f})")
        print()
        print("    - 如果已有持仓: 继续持有")
        print("    - 如果没有持仓: 关注金叉信号")
    elif '偏卖' in signal:
        print("\n>>> 建议操作: 注意风险")
        print(f"    K < D，K在较高位置 ({k_val:.2f})")
        print()
        print("    - 如果已有持仓: 注意风险控制")
        print("    - 如果没有持仓: 暂时观望")
    else:
        print("\n>>> 建议操作: 暂时观望，不操作")
        print(f"    KDJ 在中性区域")
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
        'annual': annual_return if 'annual_return' in dir() else 0,
        'current_signal': current_signal,
        'trend': trend_info,
        'current_position': current_position,
        'current_shares': current_position
    }


if __name__ == "__main__":
    run()