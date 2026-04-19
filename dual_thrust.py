import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data

INITIAL_CAPITAL = 100000
SYMBOL = 'sh.601138'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

def dual_thrust_strategy(df, k=0.5):
    """Dual Thrust 策略
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        k: 突破系数
    
    Returns:
        包含交易信号的 DataFrame
    """
    records = []
    
    for i in range(20, len(df)):
        recent = df.iloc[i-20:i]
        
        hh = recent['high'].max()
        hc = recent['close'].max()
        ll = recent['low'].min()
        lc = recent['close'].min()
        
        range_val = max(hh - lc, hc - ll)
        
        upper = df['close'].iloc[i-1] + k * range_val
        lower = df['close'].iloc[i-1] - k * range_val
        
        if df['close'].iloc[i] > upper:
            signal = 'buy'
        elif df['close'].iloc[i] < lower:
            signal = 'sell'
        else:
            signal = 'hold'
        
        records.append({
            'date': df.index[i],
            'close': df['close'].iloc[i],
            'upper': upper,
            'lower': lower,
            'signal': signal
        })
    
    return pd.DataFrame(records).set_index('date')

def get_current_signal(df, k=0.5):
    """获取当前买卖信号
    
    Args:
        df: 价格数据
        k: 突破系数
    
    Returns:
        dict: 包含信号和相关信息
    """
    if len(df) < 20:
        return {'signal': '数据不足', 'confidence': 0}
    
    recent = df.iloc[-20:]
    hh = recent['high'].max()
    hc = recent['close'].max()
    ll = recent['low'].min()
    lc = recent['close'].min()
    
    range_val = max(hh - lc, hc - ll)
    upper = df['close'].iloc[-1] + k * range_val
    lower = df['close'].iloc[-1] - k * range_val
    
    current_close = df['close'].iloc[-1]
    
    # 计算价格在通道中的位置 (0-100%)
    channel_position = (current_close - lower) / (upper - lower) * 100 if upper != lower else 50
    
    # 计算趋势
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    is_uptrend = ma5 > ma10 > ma20
    is_downtrend = ma5 < ma10 < ma20
    
    # 计算近期涨跌幅
    change_5d = (current_close / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    
    # 计算信号强度
    if current_close > upper:
        signal = '买入'
        breakout = (current_close - upper) / upper * 100
        confidence = min(100, 70 + breakout * 5)
    elif current_close < lower:
        signal = '卖出'
        breakdown = (lower - current_close) / lower * 100
        confidence = min(100, 70 + breakdown * 5)
    else:
        # 根据价格在通道中的位置和趋势判断倾向
        if channel_position > 60 or (channel_position > 40 and is_uptrend):
            signal = '观望(偏买)'
            confidence = min(100, 55 + channel_position * 0.3 + (10 if is_uptrend else 0))
        elif channel_position < 40 or (channel_position < 60 and is_downtrend):
            signal = '观望(偏卖)'
            confidence = min(100, 55 + (100 - channel_position) * 0.3 + (10 if is_downtrend else 0))
        else:
            signal = '观望'
            confidence = 50
    
    return {
        'signal': signal,
        'confidence': confidence,
        'upper': upper,
        'lower': lower,
        'current_price': current_close,
        'range': range_val,
        'channel_position': channel_position
    }

def analyze_trend(df):
    """分析趋势
    
    Returns:
        dict: 趋势信息
    """
    # 短期均线
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    current = df['close'].iloc[-1]
    
    # 趋势判断
    if ma5 > ma10 > ma20:
        trend = '上升趋势'
        trend_strength = '强'
    elif ma5 > ma10:
        trend = '上升趋势'
        trend_strength = '中'
    elif ma5 < ma10 < ma20:
        trend = '下降趋势'
        trend_strength = '强'
    elif ma5 < ma10:
        trend = '下降趋势'
        trend_strength = '中'
    else:
        trend = '横盘整理'
        trend_strength = '弱'
    
    # 近期涨跌幅
    change_5d = (current / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
    change_20d = (current / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
    
    return {
        'trend': trend,
        'strength': trend_strength,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'change_5d': change_5d,
        'change_20d': change_20d
    }

def run(symbol=None, start_date=None, end_date=None, capital=None, k=0.5):
    """运行 Dual Thrust 策略回测 (含分红)
    
    Args:
        symbol: 股票代码，默认使用全局 SYMBOL
        start_date: 开始日期
        end_date: 结束日期
        capital: 初始资金
        k: 突破系数
    
    Returns:
        回测结果字典
    """
    symbol = symbol or SYMBOL
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    
    print("=" * 60)
    print("Dual Thrust Strategy (含分红计算)")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print(f"Period: {start_date} - {end_date}")
    print(f"Initial: {capital}")
    print(f"K parameter: {k}")
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
    
    # 确保有必要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: 缺少列 {col}")
            return None
    
    print(f"\nData loaded: {len(df)} records")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    if not dividend_df.empty:
        print(f"Dividend records: {len(dividend_df)}")
    
    signals = dual_thrust_strategy(df, k)
    
    cash = capital
    position = 0  # 持仓股数
    total_dividend = 0  # 累计分红收入
    trades = []
    
    # 添加分红信息到价格数据
    if not dividend_df.empty and 'ex_date' in dividend_df.columns:
        # 转换 ex_date 为 datetime
        dividend_df['ex_date'] = pd.to_datetime(dividend_df['ex_date'])
        
        # 使用 ex_date 作为股权登记日，dividend_per_share 作为每股分红
        div_dict = {}
        for _, row in dividend_df.iterrows():
            if pd.notna(row['ex_date']) and row.get('dividend_per_share', 0) > 0:
                div_dict[row['ex_date']] = row['dividend_per_share']
    else:
        div_dict = {}
    
    # 创建持仓状态跟踪 (包含所有日期)
    position_tracker = []
    current_position = 0
    
    # 遍历所有交易日，跟踪持仓和分红
    signal_dates = set(signals.index)
    
    for date in df.index:
        # 检查是否有分红 (登记日)
        if date in div_dict and current_position > 0:
            div_per_share = div_dict[date]
            div_income = current_position * div_per_share
            cash += div_income
            total_dividend += div_income
            trades.append(('DIVIDEND', date, div_income, current_position))
        
        # 检查是否有交易信号
        if date in signal_dates:
            signal_row = signals.loc[date]
            if signal_row['signal'] == 'buy' and current_position == 0:
                units = cash / signal_row['close']
                current_position = units
                cash = 0
                trades.append(('BUY', date, signal_row['close'], current_position))
            elif signal_row['signal'] == 'sell' and current_position > 0:
                value = current_position * signal_row['close']
                cash = value
                trades.append(('SELL', date, signal_row['close'], current_position))
                current_position = 0
        
        position_tracker.append({
            'date': date,
            'position': current_position,
            'cash': cash
        })
    
    position_df = pd.DataFrame(position_tracker).set_index('date')
    
    # 最后清仓或计算最终价值
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
    
    current_signal = get_current_signal(df, k)
    trend_info = analyze_trend(df)
    
    current_price = current_signal['current_price']
    print(f"\n当前价格: {current_price:.2f}")
    print(f"趋势判断: {trend_info['trend']} (强度: {trend_info['strength']})")
    print(f"5日涨跌: {trend_info['change_5d']:+.2f}%")
    print(f"20日涨跌: {trend_info['change_20d']:+.2f}%")
    
    print(f"\n突破通道:")
    print(f"  上轨: {current_signal['upper']:.2f} (突破买入)")
    print(f"  下轨: {current_signal['lower']:.2f} (跌破卖出)")
    print(f"  区间: {current_signal['range']:.2f}")
    
    # 买卖建议
    signal = current_signal['signal']
    confidence = current_signal['confidence']
    channel_pos = current_signal.get('channel_position', 50)
    
    print(f"\n【当前信号】: {signal}")
    print(f"【信号强度】: {confidence:.0f}%")
    print(f"【通道位置】: {channel_pos:.0f}% (0%=下轨, 100%=上轨)")
    
    if signal == '买入':
        print("\n>>> 建议操作: 买入")
        print(f"    当前价格 {current_price:.2f} 已突破上轨 {current_signal['upper']:.2f}")
        print(f"    参考买入价: {current_price:.2f} 附近")
        print(f"    止损参考: {current_signal['lower']:.2f} (下轨)")
    elif signal == '卖出':
        print("\n>>> 建议操作: 卖出")
        print(f"    当前价格 {current_price:.2f} 已跌破下轨 {current_signal['lower']:.2f}")
        print("    如有持仓，建议卖出或减仓")
    elif '偏买' in signal:
        dist_upper = (current_signal['upper'] - current_price) / current_price * 100
        print("\n>>> 建议操作: 关注买入机会")
        print(f"    价格接近上轨，通道位置 {channel_pos:.0f}%")
        print(f"    距离上轨: +{dist_upper:.2f}%")
        print()
        print("    - 如果已有持仓: 继续持有")
        print("    - 如果没有持仓: 关注突破，可提前埋单")
    elif '偏卖' in signal:
        dist_lower = (current_price - current_signal['lower']) / current_price * 100
        print("\n>>> 建议操作: 注意风险")
        print(f"    价格接近下轨，通道位置 {channel_pos:.0f}%")
        print(f"    距离下轨: -{dist_lower:.2f}%")
        print()
        print("    - 如果已有持仓: 注意止损，防范跌破风险")
        print("    - 如果没有持仓: 暂时观望")
    else:
        dist_upper = (current_signal['upper'] - current_price) / current_price * 100
        dist_lower = (current_price - current_signal['lower']) / current_price * 100
        
        print("\n>>> 建议操作: 暂时观望，不操作")
        print(f"    当前价格 {current_price:.2f} 在通道中间")
        print(f"    需上涨 +{dist_upper:.2f}% 突破上轨才买入")
        print(f"    需下跌 -{dist_lower:.2f}% 跌破下轨才卖出")
        print()
        print("    - 如果已有持仓: 继续持有，等待卖出信号")
        print("    - 如果没有持仓: 暂时不买，等待突破信号")
    
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