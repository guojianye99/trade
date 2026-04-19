import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

short = 5
long = 20

def ma_crossover_strategy(df, short_period=5, long_period=20):
    """均线交叉策略
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        short_period: 短期均线周期
        long_period: 长期均线周期
    
    Returns:
        包含交易信号的 DataFrame
    """
    df = df.copy()
    df['ma_short'] = df['close'].rolling(short_period).mean()
    df['ma_long'] = df['close'].rolling(long_period).mean()
    
    df['prev_short'] = df['ma_short'].shift(1)
    df['prev_long'] = df['ma_long'].shift(1)
    
    df['position'] = 'hold'
    df.loc[(df['ma_short'] > df['ma_long']) & (df['prev_short'] <= df['prev_long']), 'position'] = 'buy'
    df.loc[(df['ma_short'] < df['ma_long']) & (df['prev_short'] >= df['prev_long']), 'position'] = 'sell'
    
    records = []
    for i in range(long_period, len(df)):
        if df['position'].iloc[i] != 'hold':
            records.append({
                'date': df.index[i],
                'close': df['close'].iloc[i],
                'ma_short': df['ma_short'].iloc[i],
                'ma_long': df['ma_long'].iloc[i],
                'position': df['position'].iloc[i]
            })
    
    return pd.DataFrame(records).set_index('date'), df

def get_current_signal(df, short_period=5, long_period=20):
    """获取当前买卖信号
    
    Returns:
        dict: 包含信号和相关信息
    """
    if len(df) < long_period:
        return {'signal': '数据不足', 'confidence': 0}
    
    ma_short = df['close'].rolling(short_period).mean()
    ma_long = df['close'].rolling(long_period).mean()
    
    current_short = ma_short.iloc[-1]
    current_long = ma_long.iloc[-1]
    prev_short = ma_short.iloc[-2]
    prev_long = ma_long.iloc[-2]
    
    current_price = df['close'].iloc[-1]
    
    # 计算均线距离百分比
    ma_diff = (current_short - current_long) / current_long * 100
    
    # 计算价格相对于均线的位置
    price_to_short = (current_price - current_short) / current_short * 100
    price_to_long = (current_price - current_long) / current_long * 100
    
    # 判断信号
    if current_short > current_long and prev_short <= prev_long:
        signal = '买入'  # 金叉刚刚形成
        confidence = min(100, 70 + abs(ma_diff) * 3)
    elif current_short < current_long and prev_short >= prev_long:
        signal = '卖出'  # 死叉刚刚形成
        confidence = min(100, 70 + abs(ma_diff) * 3)
    else:
        # 没有交叉，判断趋势强度
        if current_short > current_long:
            # 金叉状态，判断是否有买入价值
            if ma_diff > 3:
                signal = '观望(偏买)'
                confidence = min(100, 55 + ma_diff)
            elif ma_diff > 1:
                signal = '观望(偏买)'
                confidence = 60
            else:
                signal = '观望'
                confidence = 55
        else:
            # 死叉状态，判断风险
            if ma_diff < -3:
                signal = '观望(偏卖)'
                confidence = min(100, 55 + abs(ma_diff))
            elif ma_diff < -1:
                signal = '观望(偏卖)'
                confidence = 60
            else:
                signal = '观望'
                confidence = 55
    
    return {
        'signal': signal,
        'confidence': confidence,
        'ma_short': current_short,
        'ma_long': current_long,
        'ma_diff': ma_diff,
        'current_price': current_price,
        'is_golden_cross': current_short > current_long,
        'price_to_short': price_to_short,
        'price_to_long': price_to_long
    }

def analyze_trend(df):
    """分析趋势"""
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1] if len(df) > 60 else None
    current = df['close'].iloc[-1]
    
    # 趋势判断
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
    
    # 近期涨跌幅
    change_5d = (current / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    change_20d = (current / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
    
    return {
        'trend': trend,
        'strength': strength,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'ma60': ma60,
        'change_5d': change_5d,
        'change_20d': change_20d
    }

def run(symbol=None, start_date=None, end_date=None, capital=None, short_period=None, long_period=None):
    """运行均线交叉策略回测 (含分红)"""
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    short_period = short_period or short
    long_period = long_period or long
    
    print("=" * 60)
    print("Moving Average Crossover Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"Short MA: {short_period} days")
    print(f"Long MA: {long_period} days")
    print(f"Initial: {capital}")
    print("=" * 60)
    
    # 从本地加载或下载数据
    df = load_stock_data(symbol, start_date, end_date)
    if df is None or len(df) == 0:
        print("错误: 无法获取数据")
        return None
    
    # 加载分红数据
    start_year = start_date[:4]
    end_year = end_date[:4]
    dividend_df = load_dividend_data(symbol, start_year, end_year)
    
    print(f"\nData loaded: {len(df)} records")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    if not dividend_df.empty:
        print(f"Dividend records: {len(dividend_df)}")
    
    signals, df_full = ma_crossover_strategy(df, short_period, long_period)
    
    cash = capital
    position = 0
    total_dividend = 0
    trades = []
    
    # 创建分红查找字典
    if not dividend_df.empty and 'ex_date' in dividend_df.columns:
        dividend_df['ex_date'] = pd.to_datetime(dividend_df['ex_date'])
        div_dict = {}
        for _, row in dividend_df.iterrows():
            if pd.notna(row['ex_date']) and row.get('dividend_per_share', 0) > 0:
                div_dict[row['ex_date']] = row['dividend_per_share']
    else:
        div_dict = {}
    
    # 创建持仓状态跟踪
    signal_dates = set(signals.index)
    current_position = 0
    
    for date in df.index:
        # 检查分红
        if date in div_dict and current_position > 0:
            div_per_share = div_dict[date]
            div_income = current_position * div_per_share
            cash += div_income
            total_dividend += div_income
            trades.append(('DIVIDEND', date, div_income, current_position))
        
        # 检查交易信号
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
    
    current_signal = get_current_signal(df, short_period, long_period)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\n均线状态:")
    print(f"  MA{short_period}: {current_signal['ma_short']:.2f}")
    print(f"  MA{long_period}: {current_signal['ma_long']:.2f}")
    print(f"  偏离度: {current_signal['ma_diff']:+.2f}%")
    
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    is_golden = current_signal['is_golden_cross']
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    MA{short_period}({current_signal['ma_short']:.2f}) 上穿 MA{long_period}({current_signal['ma_long']:.2f})，形成金叉")
        print(f"    参考买入价: {current_price:.2f} 附近")
        print(f"    止损参考: {current_signal['ma_long']:.2f} (长期均线)")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    MA{short_period}({current_signal['ma_short']:.2f}) 下穿 MA{long_period}({current_signal['ma_long']:.2f})，形成死叉")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    当前处于金叉状态，MA{short_period} > MA{long_period}")
        print(f"    均线偏离: +{abs(current_signal['ma_diff']):.2f}%")
        print()
        print("    - 如果已有持仓: 继续持有，享受上涨")
        print("    - 如果没有持仓: 可在回调时考虑买入")
    elif '偏卖' in signal:
        print("\n>>> 建议操作: 注意风险")
        print(f"    当前处于死叉状态，MA{short_period} < MA{long_period}")
        print(f"    均线偏离: -{abs(current_signal['ma_diff']):.2f}%")
        print()
        print("    - 如果已有持仓: 注意止损，防范继续下跌")
        print("    - 如果没有持仓: 暂时观望，等待趋势好转")
    else:
        print("\n>>> 建议操作: 暂时观望，不操作")
        if is_golden:
            print(f"    当前 MA{short_period}({current_signal['ma_short']:.2f}) > MA{long_period}({current_signal['ma_long']:.2f})，金叉初期")
            print(f"    偏离度: +{abs(current_signal['ma_diff']):.2f}%")
            print()
            print("    - 如果已有持仓: 继续持有，等待死叉卖出信号")
            print("    - 如果没有持仓: 可考虑小仓位试探")
        else:
            print(f"    当前 MA{short_period}({current_signal['ma_short']:.2f}) < MA{long_period}({current_signal['ma_long']:.2f})，死叉初期")
            print(f"    偏离度: -{abs(current_signal['ma_diff']):.2f}%")
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