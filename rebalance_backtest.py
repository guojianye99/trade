import numpy as np
import pandas as pd
from datetime import datetime
from data_manager import load_stock_data, load_dividend_data, download_multiple

INITIAL_CAPITAL = 100000
SYMBOLS = ['sh.600036', 'sh.601318']  # 招商银行 + 中国平安
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

def get_monthly_index(price):
    """获取月度再平衡索引"""
    idx = []
    prev_month = None
    for i, d in enumerate(price.index):
        month = d.to_period('M')
        if month != prev_month:
            idx.append(i)
            prev_month = month
    return idx

def get_quarterly_index(price):
    """获取季度再平衡索引"""
    idx = []
    prev_quarter = None
    for i, d in enumerate(price.index):
        quarter = d.to_period('Q')
        if quarter != prev_quarter:
            idx.append(i)
            prev_quarter = quarter
    return idx

def single_invest(price, initial, dividend_dicts):
    """单资产投资 (含分红)"""
    result = []
    for col_idx, col in enumerate(price.columns):
        start_price = price[col].iloc[0]
        units = initial / start_price
        end_price = price[col].iloc[-1]
        
        total_div = 0
        div_dict = dividend_dicts[col_idx] if col_idx < len(dividend_dicts) else {}
        
        for date, div_per_share in div_dict.items():
            if date in price.index:
                total_div += units * div_per_share
        
        end_value = units * end_price + total_div
        
        result.append({
            'symbol': col,
            'end_value': end_value,
            'dividend': total_div,
            'return': (end_value - initial) / initial * 100
        })
    return result

def rebalance_strategy(price, initial, dividend_dicts, freq='monthly'):
    """再平衡策略 (含分红)"""
    close = price.values
    dates = price.index
    
    if freq == 'quarterly':
        rebal_idx = get_quarterly_index(price)
        label = "Quarterly"
    else:
        rebal_idx = get_monthly_index(price)
        label = "Monthly"
    
    print(f"\n{label} rebalance times: {len(rebal_idx)}")
    
    cash = initial
    holdings = np.array([0.0, 0.0])
    total_dividend = 0
    
    history = []
    
    for idx in rebal_idx:
        current_price = close[idx]
        current_date = dates[idx]
        
        for col_idx in range(2):
            div_dict = dividend_dicts[col_idx] if col_idx < len(dividend_dicts) else {}
            if current_date in div_dict and holdings[col_idx] > 0:
                div_income = holdings[col_idx] * div_dict[current_date]
                cash += div_income
                total_dividend += div_income
        
        asset_value = holdings * current_price
        total = cash + asset_value.sum()
        
        target = total / 2
        diff = target - asset_value
        
        for j in range(2):
            if diff[j] > 0 and current_price[j] > 0:
                buy_units = diff[j] / current_price[j]
                cost = buy_units * current_price[j]
                if cost <= cash:
                    holdings[j] += buy_units
                    cash -= cost
            elif diff[j] < 0 and current_price[j] > 0:
                sell_value = -diff[j]
                if sell_value <= asset_value[j]:
                    sell_units = sell_value / current_price[j]
                    holdings[j] -= sell_units
                    cash += sell_value
        
        portfolio_value = cash + (holdings * current_price).sum()
        history.append({
            'date': current_date,
            'cash': cash,
            'total': portfolio_value,
            'dividend': total_dividend
        })
    
    return pd.DataFrame(history).set_index('date')

def calc_metrics(history, initial):
    """计算策略指标"""
    total_value = history['total']
    ret = total_value.pct_change(fill_method=None).dropna()
    
    total_ret = (total_value.iloc[-1] - initial) / initial * 100
    
    first_date = history.index[0]
    last_date = history.index[-1]
    years = (last_date - first_date).days / 365
    
    annual_ret = ((1 + total_ret/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    
    cum = (1 + ret).cumprod()
    run_max = cum.cummax()
    dd = (cum - run_max) / run_max
    max_dd = dd.min() * 100
    
    return {
        'total_return': total_ret,
        'annual': annual_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final': total_value.iloc[-1]
    }

def analyze_asset(price, symbol, dividend_dict):
    """分析单个资产"""
    code = symbol.split('.')[1]
    current_price = price[code].iloc[-1]
    
    # 计算均线
    ma5 = price[code].rolling(5).mean().iloc[-1]
    ma10 = price[code].rolling(10).mean().iloc[-1]
    ma20 = price[code].rolling(20).mean().iloc[-1]
    
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
    
    # 近期涨跌
    change_5d = (current_price / price[code].iloc[-6] - 1) * 100 if len(price) > 5 else 0
    change_20d = (current_price / price[code].iloc[-21] - 1) * 100 if len(price) > 20 else 0
    
    # 信号判断
    if ma5 > ma20:
        signal = '偏多'
    elif ma5 < ma20:
        signal = '偏空'
    else:
        signal = '中性'
    
    return {
        'code': code,
        'current_price': current_price,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'trend': trend,
        'trend_strength': trend_strength,
        'change_5d': change_5d,
        'change_20d': change_20d,
        'signal': signal
    }

def run(symbols=None, start_date=None, end_date=None, capital=None):
    """运行再平衡回测 (含分红)"""
    symbols = symbols or SYMBOLS
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    capital = capital or INITIAL_CAPITAL
    
    print("=" * 60)
    print("Fund Rebalance Backtest (含分红计算)")
    print("=" * 60)
    print(f"Initial: {capital} CNY")
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} - {end_date}")
    print("=" * 60)
    
    data_dict = {}
    dividend_dicts = []
    
    for symbol in symbols:
        df = load_stock_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            code = symbol.split('.')[1]
            data_dict[code] = df['close']
            print(f"加载 {symbol}: {len(df)} 条记录")
            
            start_year = start_date[:4]
            end_year = end_date[:4]
            div_df = load_dividend_data(symbol, start_year, end_year)
            
            div_dict = {}
            if not div_df.empty and 'ex_date' in div_df.columns:
                div_df['ex_date'] = pd.to_datetime(div_df['ex_date'])
                for _, row in div_df.iterrows():
                    if pd.notna(row['ex_date']) and row.get('dividend_per_share', 0) > 0:
                        div_dict[row['ex_date']] = row['dividend_per_share']
                
                if div_dict:
                    print(f"  分红记录: {len(div_dict)} 次")
            
            dividend_dicts.append(div_dict)
    
    if len(data_dict) < 2:
        print("错误: 需要至少两只股票的数据")
        return None
    
    price = pd.DataFrame(data_dict)
    price = price.dropna()
    
    print(f"\n合并后数据: {len(price)} 条记录")
    print(f"Period: {price.index[0].date()} to {price.index[-1].date()}")
    
    single_result = single_invest(price, capital, dividend_dicts)
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print("\nStrategy 1: Single " + symbols[0].split('.')[1])
    print("-" * 40)
    s1 = single_result[0]['return']
    print(f"End Value: {single_result[0]['end_value']:,.2f}")
    print(f"Dividend: {single_result[0]['dividend']:,.2f}")
    print(f"Return: {s1:.2f}%")
    
    print("\nStrategy 2: Single " + symbols[1].split('.')[1])
    print("-" * 40)
    s2 = single_result[1]['return']
    print(f"End Value: {single_result[1]['end_value']:,.2f}")
    print(f"Dividend: {single_result[1]['dividend']:,.2f}")
    print(f"Return: {s2:.2f}%")
    
    history_monthly = rebalance_strategy(price, capital, dividend_dicts, freq='monthly')
    m_monthly = calc_metrics(history_monthly, capital)
    
    print("\nStrategy 3: Monthly Rebalance")
    print("-" * 40)
    print(f"Final: {m_monthly['final']:,.2f}")
    print(f"Total: {m_monthly['total_return']:.2f}%")
    print(f"Annual: {m_monthly['annual']:.2f}%")
    print(f"Sharpe: {m_monthly['sharpe']:.2f}")
    print(f"Max DD: {m_monthly['max_dd']:.2f}%")
    
    history_quarterly = rebalance_strategy(price, capital, dividend_dicts, freq='quarterly')
    m_quarterly = calc_metrics(history_quarterly, capital)
    
    print("\nStrategy 4: Quarterly Rebalance")
    print("-" * 40)
    print(f"Final: {m_quarterly['final']:,.2f}")
    print(f"Total: {m_quarterly['total_return']:.2f}%")
    print(f"Annual: {m_quarterly['annual']:.2f}%")
    print(f"Sharpe: {m_quarterly['sharpe']:.2f}")
    print(f"Max DD: {m_quarterly['max_dd']:.2f}%")
    
    # 当前买卖建议
    print(f"\n" + "=" * 60)
    print("当前资产分析与买卖建议")
    print("=" * 60)
    
    # 分析每个资产
    asset_analyses = []
    for idx, symbol in enumerate(symbols):
        analysis = analyze_asset(price, symbol, dividend_dicts[idx])
        asset_analyses.append(analysis)
        
        print(f"\n【{analysis['code']}】")
        print(f"  当前价格: {analysis['current_price']:.2f}")
        print(f"  趋势判断: {analysis['trend']} (强度: {analysis['trend_strength']})")
        print(f"  5日涨跌: {analysis['change_5d']:+.2f}%")
        print(f"  20日涨跌: {analysis['change_20d']:+.2f}%")
        print(f"  MA5: {analysis['ma5']:.2f}")
        print(f"  MA10: {analysis['ma10']:.2f}")
        print(f"  MA20: {analysis['ma20']:.2f}")
        print(f"  信号: {analysis['signal']}")
    
    # 综合建议
    print(f"\n" + "-" * 40)
    print("【综合建议】")
    
    signals = [a['signal'] for a in asset_analyses]
    trends = [a['trend'] for a in asset_analyses]
    
    # 判断当前整体状态
    overall_signal = '观望'
    
    # 判断是否适合再平衡
    if all('上升' in t for t in trends):
        overall_signal = '持有'
        print("\n>>> 当前状态: 两只资产均处于上升趋势")
        print()
        print("    - 如果已有持仓: 继续持有，保持再平衡策略")
        print("    - 如果没有持仓: 可以考虑建仓，分批买入")
    elif all('下降' in t for t in trends):
        overall_signal = '谨慎'
        print("\n>>> 当前状态: 两只资产均处于下降趋势，注意风险")
        print()
        print("    - 如果已有持仓: 考虑减仓或止损")
        print("    - 如果没有持仓: 暂时观望，等待趋势好转")
    elif '上升' in trends[0] and '下降' in trends[1]:
        print(f"\n>>> 当前状态: {asset_analyses[0]['code']} 上升，{asset_analyses[1]['code']} 下降")
        print()
        print(f"    - {asset_analyses[0]['code']} 走势较强，可考虑增加配置")
        print(f"    - {asset_analyses[1]['code']} 走势较弱，可考虑减少配置")
    elif '下降' in trends[0] and '上升' in trends[1]:
        print(f"\n>>> 当前状态: {asset_analyses[0]['code']} 下降，{asset_analyses[1]['code']} 上升")
        print()
        print(f"    - {asset_analyses[0]['code']} 走势较弱，可考虑减少配置")
        print(f"    - {asset_analyses[1]['code']} 走势较强，可考虑增加配置")
    else:
        print("\n>>> 当前状态: 资产走势分化，建议观望")
        print()
        print("    - 如果已有持仓: 维持当前配置")
        print("    - 如果没有持仓: 暂时观望，等待明确信号")
    
    print(f"\n【综合信号】: {overall_signal}")
    
    # 对比总结
    print("\n" + "=" * 60)
    print("策略对比")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Total':>10} {'Annual':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 60)
    print(f"{'Single ' + symbols[0].split('.')[1]:<20} {s1:>9.2f}% {'--':>10} {'--':>8} {'--':>10}")
    print(f"{'Single ' + symbols[1].split('.')[1]:<20} {s2:>9.2f}% {'--':>10} {'--':>8} {'--':>10}")
    print(f"{'Monthly Rebalance':<20} {m_monthly['total_return']:>9.2f}% {m_monthly['annual']:>9.2f}% {m_monthly['sharpe']:>7.2f} {m_monthly['max_dd']:>9.2f}%")
    print(f"{'Quarterly Rebalance':<20} {m_quarterly['total_return']:>9.2f}% {m_quarterly['annual']:>9.2f}% {m_quarterly['sharpe']:>7.2f} {m_quarterly['max_dd']:>9.2f}%")
    
    best = max(s1, s2, m_monthly['total_return'], m_quarterly['total_return'])
    print("-" * 60)
    if best == s1:
        print(f"Best: Single {symbols[0].split('.')[1]}")
    elif best == s2:
        print(f"Best: Single {symbols[1].split('.')[1]}")
    elif best == m_monthly['total_return']:
        print("Best: Monthly Rebalance")
    else:
        print("Best: Quarterly Rebalance")
    print("=" * 60)
    
    return {
        'single1': s1,
        'single2': s2,
        'monthly': m_monthly,
        'quarterly': m_quarterly,
        'asset_analyses': asset_analyses
    }

if __name__ == "__main__":
    run()