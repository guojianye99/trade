import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

fast = 12
slow = 26
signal_period_default = 9

def macd_strategy(df, fast_period=12, slow_period=26, signal_period=9):
    """MACD 策略"""
    df = df.copy()
    
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal_line
    
    df['macd'] = macd
    df['signal_line'] = signal_line
    df['histogram'] = histogram
    
    df['prev_macd'] = df['macd'].shift(1)
    df['prev_signal'] = df['signal_line'].shift(1)
    
    df['position'] = 'hold'
    df.loc[(df['macd'] > df['signal_line']) & (df['prev_macd'] <= df['prev_signal']), 'position'] = 'buy'
    df.loc[(df['macd'] < df['signal_line']) & (df['prev_macd'] >= df['prev_signal']), 'position'] = 'sell'
    
    records = []
    for i in range(slow_period + signal_period, len(df)):
        if df['position'].iloc[i] != 'hold':
            records.append({
                'date': df.index[i],
                'close': df['close'].iloc[i],
                'macd': df['macd'].iloc[i],
                'signal': df['signal_line'].iloc[i],
                'position': df['position'].iloc[i]
            })
    
    return pd.DataFrame(records).set_index('date'), df

def get_current_signal(df, fast_period=12, slow_period=26, signal_period=9):
    """获取当前买卖信号"""
    if len(df) < slow_period + signal_period:
        return {'signal': '数据不足', 'confidence': 0}
    
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal_line
    
    current_macd = macd.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_hist = histogram.iloc[-1]
    prev_macd = macd.iloc[-2]
    prev_signal = signal_line.iloc[-2]
    
    current_price = df['close'].iloc[-1]
    
    # 计算 histogram 相对强度
    hist_strength = abs(current_hist) / abs(current_signal) * 100 if current_signal != 0 else 0
    
    # 判断信号
    if current_macd > current_signal and prev_macd <= prev_signal:
        signal = '买入'  # MACD金叉刚刚形成
        confidence = min(100, 70 + abs(current_hist) * 20)
    elif current_macd < current_signal and prev_macd >= prev_signal:
        signal = '卖出'  # MACD死叉刚刚形成
        confidence = min(100, 70 + abs(current_hist) * 20)
    else:
        # 没有交叉，判断趋势强度
        if current_macd > current_signal:
            # 金叉状态
            if current_hist > 0 and hist_strength > 5:
                signal = '观望(偏买)'
                confidence = min(100, 55 + hist_strength)
            else:
                signal = '观望'
                confidence = 60
        else:
            # 死叉状态
            if current_hist < 0 and hist_strength > 5:
                signal = '观望(偏卖)'
                confidence = min(100, 55 + hist_strength)
            else:
                signal = '观望'
                confidence = 60
    
    return {
        'signal': signal,
        'confidence': confidence,
        'macd': current_macd,
        'signal_line': current_signal,
        'histogram': current_hist,
        'hist_strength': hist_strength,
        'current_price': current_price,
        'is_golden_cross': current_macd > current_signal
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
        fast_period=None, slow_period=None, signal_period=None):
    """运行 MACD 策略回测 (含分红)"""
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    fast_period = fast_period or fast
    slow_period = slow_period or slow
    signal_period = signal_period or signal_period_default
    
    print("=" * 60)
    print("MACD Trend Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"Fast EMA: {fast_period}")
    print(f"Slow EMA: {slow_period}")
    print(f"Signal: {signal_period}")
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
    
    signals, df_full = macd_strategy(df, fast_period, slow_period, signal_period)
    
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
    
    current_signal = get_current_signal(df, fast_period, slow_period, signal_period)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\nMACD指标:")
    print(f"  MACD: {current_signal['macd']:.4f}")
    print(f"  Signal: {current_signal['signal_line']:.4f}")
    print(f"  Histogram: {current_signal['histogram']:.4f}")
    
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    is_golden = current_signal['is_golden_cross']
    hist_strength = current_signal.get('hist_strength', 0)
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    print(f"【动能强度】: {hist_strength:.1f}%")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    MACD({current_signal['macd']:.4f}) 上穿 Signal({current_signal['signal_line']:.4f})，形成金叉")
        print(f"    参考买入价: {current_price:.2f} 附近")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    MACD({current_signal['macd']:.4f}) 下穿 Signal({current_signal['signal_line']:.4f})，形成死叉")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    MACD > Signal，金叉状态，多头动能较强")
        print(f"    Histogram: +{abs(current_signal['histogram']):.4f}")
        print()
        print("    - 如果已有持仓: 继续持有，享受上涨")
        print("    - 如果没有持仓: 可在回调时考虑买入")
    elif '偏卖' in signal:
        print("\n>>> 建议操作: 注意风险")
        print(f"    MACD < Signal，死叉状态，空头动能较强")
        print(f"    Histogram: -{abs(current_signal['histogram']):.4f}")
        print()
        print("    - 如果已有持仓: 注意止损，防范继续下跌")
        print("    - 如果没有持仓: 暂时观望，等待趋势好转")
    else:
        print("\n>>> 建议操作: 暂时观望，不操作")
        if is_golden:
            print(f"    当前 MACD > Signal，金叉状态")
            print(f"    Histogram: +{abs(current_signal['histogram']):.4f} (多头动能)")
            print()
            print("    - 如果已有持仓: 继续持有，等待死叉卖出信号")
            print("    - 如果没有持仓: 可考虑小仓位试探")
        else:
            print(f"    当前 MACD < Signal，死叉状态")
            print(f"    Histogram: -{abs(current_signal['histogram']):.4f} (空头动能)")
            print()
            print("    - 如果已有持仓: 等待金叉出现")
            print("    - 如果没有持仓: 暂时不买，等待金叉信号")
    
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
        'trend': trend_info
    }

if __name__ == "__main__":
    run()