#!/usr/bin/env python3
"""
RSI 相对强弱指标策略
- RSI > 70: 超买区域，可能回调
- RSI < 30: 超卖区域，可能反弹
- RSI 50 为多空分界线
"""

import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# RSI 参数
RSI_PERIOD = 14
OVERBOUGHT = 70  # 超买线
OVERSOLD = 30    # 超卖线


def calculate_rsi(df, period=14):
    """计算 RSI 指标"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def rsi_strategy(df, period=14, overbought=70, oversold=30):
    """RSI 策略
    
    买入信号: RSI 从下往上穿过超卖线(30)
    卖出信号: RSI 从上往下穿过超买线(70)
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df, period)
    df['prev_rsi'] = df['rsi'].shift(1)
    
    df['position'] = 'hold'
    # RSI 从超卖区上穿30，买入
    df.loc[(df['rsi'] > oversold) & (df['prev_rsi'] <= oversold), 'position'] = 'buy'
    # RSI 从超买区下穿70，卖出
    df.loc[(df['rsi'] < overbought) & (df['prev_rsi'] >= overbought), 'position'] = 'sell'
    
    records = []
    for i in range(period, len(df)):
        if df['position'].iloc[i] != 'hold':
            records.append({
                'date': df.index[i],
                'close': df['close'].iloc[i],
                'rsi': df['rsi'].iloc[i],
                'position': df['position'].iloc[i]
            })
    
    # 如果没有信号，返回带正确列的空 DataFrame
    if records:
        return pd.DataFrame(records).set_index('date'), df
    else:
        return pd.DataFrame(columns=['close', 'rsi', 'position']), df


def get_current_signal(df, period=14, overbought=70, oversold=30):
    """获取当前买卖信号"""
    if len(df) < period:
        return {'signal': '数据不足', 'confidence': 0}
    
    rsi = calculate_rsi(df, period)
    current_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    current_price = df['close'].iloc[-1]
    
    # 判断信号
    if current_rsi > oversold and prev_rsi <= oversold:
        signal = '买入'
        confidence = min(100, 70 + (50 - current_rsi))
    elif current_rsi < overbought and prev_rsi >= overbought:
        signal = '卖出'
        confidence = min(100, 70 + (current_rsi - 50))
    else:
        # 根据RSI位置判断倾向
        if current_rsi > 60:
            signal = '观望(偏卖)'
            confidence = min(100, 50 + (current_rsi - 50))
        elif current_rsi < 40:
            signal = '观望(偏买)'
            confidence = min(100, 50 + (50 - current_rsi))
        else:
            signal = '观望'
            confidence = 50
    
    return {
        'signal': signal,
        'confidence': confidence,
        'rsi': current_rsi,
        'current_price': current_price,
        'overbought': overbought,
        'oversold': oversold
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
        period=None, overbought=None, oversold=None):
    """运行 RSI 策略回测"""
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    period = period or RSI_PERIOD
    overbought = overbought or OVERBOUGHT
    oversold = oversold or OVERSOLD
    
    print("=" * 60)
    print("RSI Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"RSI Period: {period}")
    print(f"Overbought: {overbought}")
    print(f"Oversold: {oversold}")
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
    
    signals, df_full = rsi_strategy(df, period, overbought, oversold)
    
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
    
    current_signal = get_current_signal(df, period, overbought, oversold)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    rsi_value = current_signal['rsi']
    
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\nRSI指标:")
    print(f"  RSI({period}): {rsi_value:.2f}")
    print(f"  超买线: {overbought}")
    print(f"  超卖线: {oversold}")
    
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    RSI 从超卖区上穿 {oversold}")
        print(f"    参考买入价: {current_price:.2f} 附近")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    RSI 从超买区下穿 {overbought}")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    RSI 在低位区域 ({rsi_value:.2f})")
        print()
        print("    - 如果已有持仓: 继续持有")
        print("    - 如果没有持仓: 关注超卖反弹机会")
    elif '偏卖' in signal:
        print("\n>>> 建议操作: 注意风险")
        print(f"    RSI 在高位区域 ({rsi_value:.2f})")
        print()
        print("    - 如果已有持仓: 注意止盈，防范回调")
        print("    - 如果没有持仓: 暂时观望")
    else:
        print("\n>>> 建议操作: 暂时观望，不操作")
        print(f"    RSI 在中性区域 ({rsi_value:.2f})")
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