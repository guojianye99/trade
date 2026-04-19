"""
动量策略模块 - 实现价格动量、行业动量、盈余动量等策略
"""
import numpy as np
import pandas as pd
from datetime import datetime


class PriceMomentum:
    """价格动量策略"""
    
    def __init__(self, lookback_period=120, holding_period=20):
        """
        初始化价格动量策略
        
        Args:
            lookback_period: 回看期间(交易日)
            holding_period: 持有期间(交易日)
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
    
    def calculate_momentum(self, close_prices):
        """
        计算动量值
        
        Args:
            close_prices: 收盘价Series
        
        Returns:
            动量值
        """
        if len(close_prices) < self.lookback_period:
            return None
        
        # 简单收益率动量
        momentum = close_prices.iloc[-1] / close_prices.iloc[-self.lookback_period] - 1
        
        return momentum
    
    def calculate_risk_adjusted_momentum(self, close_prices):
        """
        计算风险调整后动量
        
        Args:
            close_prices: 收盘价Series
        
        Returns:
            风险调整后动量
        """
        if len(close_prices) < self.lookback_period:
            return None
        
        returns = close_prices.pct_change().iloc[-self.lookback_period:]
        
        # 夏普比率形式的动量
        if returns.std() == 0:
            return 0
        
        risk_adj_momentum = returns.mean() / returns.std() * np.sqrt(252)
        
        return risk_adj_momentum
    
    def generate_signal(self, close_prices):
        """
        生成交易信号
        
        Args:
            close_prices: 收盘价Series
        
        Returns:
            信号字典
        """
        momentum = self.calculate_momentum(close_prices)
        
        if momentum is None:
            return {
                'signal': '数据不足',
                'confidence': 0,
                'momentum': 0
            }
        
        # 根据动量值判断信号
        if momentum > 0.15:  # 15%以上涨幅
            signal = '强烈买入'
            confidence = 85
        elif momentum > 0.08:
            signal = '买入'
            confidence = 75
        elif momentum > 0.02:
            signal = '偏买'
            confidence = 65
        elif momentum > -0.02:
            signal = '观望'
            confidence = 50
        elif momentum > -0.08:
            signal = '偏卖'
            confidence = 65
        else:
            signal = '卖出'
            confidence = 75
        
        return {
            'signal': signal,
            'confidence': confidence,
            'momentum': momentum
        }


class TimeSeriesMomentum:
    """时间序列动量策略"""
    
    def __init__(self, lookback=12, holding=1):
        """
        初始化时间序列动量策略
        
        Args:
            lookback: 回看期间(月)
            holding: 持有期间(月)
        """
        self.lookback = lookback
        self.holding = holding
    
    def calculate_tsm_signal(self, prices):
        """
        计算TSM信号
        
        Args:
            prices: 价格序列
        
        Returns:
            信号 (1:多头, -1:空头, 0:平仓)
        """
        if len(prices) < self.lookback * 20:  # 假设每月20个交易日
            return 0
        
        # 计算过去lookback期的收益率
        lookback_days = self.lookback * 20
        ret = prices.iloc[-1] / prices.iloc[-lookback_days] - 1
        
        # 正收益做多,负收益做空
        if ret > 0:
            return 1
        elif ret < 0:
            return -1
        else:
            return 0
    
    def backtest_tsm(self, prices, initial_capital=100000):
        """
        回测TSM策略
        
        Args:
            prices: 价格序列
            initial_capital: 初始资金
        
        Returns:
            回测结果DataFrame
        """
        lookback_days = self.lookback * 20
        holding_days = self.holding * 20
        
        if len(prices) < lookback_days + holding_days:
            return pd.DataFrame()
        
        capital = initial_capital
        results = []
        
        for i in range(lookback_days, len(prices) - holding_days, holding_days):
            # 计算信号
            past_prices = prices.iloc[i-lookback_days:i]
            signal = self.calculate_tsm_signal(past_prices)
            
            # 计算收益
            entry_price = prices.iloc[i]
            exit_price = prices.iloc[i + holding_days]
            
            ret = (exit_price / entry_price - 1) * signal
            
            capital *= (1 + ret)
            
            results.append({
                'date': prices.index[i],
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': ret,
                'capital': capital
            })
        
        return pd.DataFrame(results)


class CrossSectionalMomentum:
    """横截面动量策略"""
    
    def __init__(self, lookback=120, top_pct=0.3, bottom_pct=0.3):
        """
        初始化横截面动量策略
        
        Args:
            lookback: 回看期间(交易日)
            top_pct: 买入前%股票
            bottom_pct: 卖出后%股票
        """
        self.lookback = lookback
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
    
    def rank_stocks(self, price_data):
        """
        根据动量对股票排序
        
        Args:
            price_data: 价格数据DataFrame (index为日期, columns为股票代码)
        
        Returns:
            排序后的股票列表
        """
        if len(price_data) < self.lookback:
            return []
        
        # 计算各股票动量
        momentum = price_data.iloc[-1] / price_data.iloc[-self.lookback] - 1
        
        # 排序
        ranked = momentum.sort_values(ascending=False)
        
        return ranked
    
    def select_portfolio(self, price_data):
        """
        选择投资组合
        
        Args:
            price_data: 价格数据DataFrame
        
        Returns:
            (多头组合, 空头组合)
        """
        ranked = self.rank_stocks(price_data)
        
        if len(ranked) == 0:
            return [], []
        
        n_stocks = len(ranked)
        top_n = int(n_stocks * self.top_pct)
        bottom_n = int(n_stocks * self.bottom_pct)
        
        long_portfolio = ranked.head(top_n).index.tolist()
        short_portfolio = ranked.tail(bottom_n).index.tolist()
        
        return long_portfolio, short_portfolio
    
    def calculate_portfolio_return(self, price_data, holding_period=20):
        """
        计算组合收益
        
        Args:
            price_data: 价格数据DataFrame
            holding_period: 持有期间
        
        Returns:
            组合收益率
        """
        long_portfolio, short_portfolio = self.select_portfolio(price_data)
        
        if not long_portfolio or len(price_data) < holding_period:
            return 0
        
        # 多头收益
        long_return = (price_data[long_portfolio].iloc[-holding_period:].pct_change() + 1).prod().mean() - 1
        
        # 空头收益(做空)
        if short_portfolio:
            short_return = (price_data[short_portfolio].iloc[-holding_period:].pct_change() + 1).prod().mean() - 1
            short_return = -short_return  # 做空收益取反
        else:
            short_return = 0
        
        # 组合收益
        portfolio_return = (long_return + short_return) / 2
        
        return portfolio_return


class IndustryMomentum:
    """行业动量策略"""
    
    def __init__(self, lookback=120):
        """
        初始化行业动量策略
        
        Args:
            lookback: 回看期间(交易日)
        """
        self.lookback = lookback
    
    def calculate_industry_momentum(self, industry_data):
        """
        计算行业动量
        
        Args:
            industry_data: 行业指数数据DataFrame
        
        Returns:
            行业动量Series
        """
        if len(industry_data) < self.lookback:
            return pd.Series()
        
        momentum = industry_data.iloc[-1] / industry_data.iloc[-self.lookback] - 1
        
        return momentum
    
    def select_top_industries(self, industry_data, n=3):
        """
        选择表现最好的行业
        
        Args:
            industry_data: 行业指数数据
            n: 选择行业数量
        
        Returns:
            行业列表
        """
        momentum = self.calculate_industry_momentum(industry_data)
        
        if momentum.empty:
            return []
        
        return momentum.nlargest(n).index.tolist()


class EarningsMomentum:
    """盈余动量策略"""
    
    def __init__(self):
        """初始化盈余动量策略"""
        pass
    
    def calculate_earnings_surprise(self, actual_eps, expected_eps):
        """
        计算盈余惊喜
        
        Args:
            actual_eps: 实际EPS
            expected_eps: 预期EPS
        
        Returns:
            盈余惊喜比例
        """
        if expected_eps == 0:
            return 0
        
        surprise = (actual_eps - expected_eps) / abs(expected_eps)
        
        return surprise
    
    def calculate_earnings_momentum(self, earnings_history):
        """
        计算盈余动量
        
        Args:
            earnings_history: EPS历史序列
        
        Returns:
            盈余动量
        """
        if len(earnings_history) < 4:
            return 0
        
        # 最近4个季度的EPS增长率
        recent_eps = earnings_history.iloc[-1]
        past_eps = earnings_history.iloc[-4]
        
        if past_eps == 0:
            return 0
        
        momentum = (recent_eps - past_eps) / abs(past_eps)
        
        return momentum
    
    def generate_signal(self, actual_eps, expected_eps, earnings_history):
        """
        生成信号
        
        Args:
            actual_eps: 实际EPS
            expected_eps: 预期EPS
            earnings_history: EPS历史
        
        Returns:
            信号字典
        """
        surprise = self.calculate_earnings_surprise(actual_eps, expected_eps)
        momentum = self.calculate_earnings_momentum(earnings_history)
        
        # 综合信号
        combined_score = surprise * 0.5 + momentum * 0.5
        
        if combined_score > 0.1:
            signal = '买入'
            confidence = 75
        elif combined_score > 0:
            signal = '偏买'
            confidence = 60
        elif combined_score > -0.1:
            signal = '偏卖'
            confidence = 60
        else:
            signal = '卖出'
            confidence = 75
        
        return {
            'signal': signal,
            'confidence': confidence,
            'earnings_surprise': surprise,
            'earnings_momentum': momentum,
            'combined_score': combined_score
        }


def get_current_momentum_signal(close_prices, lookback=120):
    """
    获取当前动量信号(用于main.py集成)
    
    Args:
        close_prices: 收盘价Series
        lookback: 回看期间
    
    Returns:
        信号字典
    """
    strategy = PriceMomentum(lookback_period=lookback)
    return strategy.generate_signal(close_prices)


def momentum_backtest(prices, initial_capital=100000, lookback=120, holding=20):
    """
    动量策略回测
    
    Args:
        prices: 价格序列
        initial_capital: 初始资金
        lookback: 回看期间
        holding: 持有期间
    
    Returns:
        回测结果
    """
    tsm = TimeSeriesMomentum(lookback=lookback//20, holding=holding//20)
    return tsm.backtest_tsm(prices, initial_capital)