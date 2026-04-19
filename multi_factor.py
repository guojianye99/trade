"""
多因子选股策略 - 基于价值、成长、质量、动量等多因子模型
"""
import numpy as np
import pandas as pd
from datetime import datetime


class FactorModel:
    """多因子模型"""
    
    def __init__(self):
        """初始化因子模型"""
        self.factors = {}
        self.weights = {}
    
    def add_factor(self, name, values, weight=1.0):
        """
        添加因子
        
        Args:
            name: 因子名称
            values: 因子值(Series, index为股票代码)
            weight: 因子权重
        """
        self.factors[name] = values
        self.weights[name] = weight
    
    def normalize_factor(self, values, method='zscore'):
        """
        因子标准化
        
        Args:
            values: 因子值
            method: 标准化方法('zscore', 'minmax', 'rank')
        
        Returns:
            标准化后的因子值
        """
        if method == 'zscore':
            return (values - values.mean()) / values.std()
        elif method == 'minmax':
            return (values - values.min()) / (values.max() - values.min())
        elif method == 'rank':
            return values.rank(pct=True)
        else:
            return values
    
    def calculate_scores(self, normalize_method='zscore'):
        """
        计算综合得分
        
        Args:
            normalize_method: 标准化方法
        
        Returns:
            综合得分Series
        """
        if not self.factors:
            return pd.Series()
        
        # 标准化所有因子
        normalized_factors = {}
        for name, values in self.factors.items():
            normalized_factors[name] = self.normalize_factor(values, normalize_method)
        
        # 加权求和
        score = pd.Series(0, index=list(self.factors.values())[0].index)
        for name, values in normalized_factors.items():
            score += values * self.weights[name]
        
        return score
    
    def select_top_stocks(self, n=10, normalize_method='zscore'):
        """
        选择得分最高的N只股票
        
        Args:
            n: 股票数量
            normalize_method: 标准化方法
        
        Returns:
            股票代码列表
        """
        scores = self.calculate_scores(normalize_method)
        return scores.nlargest(n).index.tolist()


class ValueFactor:
    """价值因子"""
    
    @staticmethod
    def pe_factor(pe_ratio):
        """
        PE因子(越低越好)
        
        Args:
            pe_ratio: 市盈率Series
        
        Returns:
            因子值(负值,因为PE越低越好)
        """
        # 取负值,因为PE越低越好
        return -pe_ratio.replace([np.inf, -np.inf], np.nan).fillna(pe_ratio.median())
    
    @staticmethod
    def pb_factor(pb_ratio):
        """
        PB因子(越低越好)
        
        Args:
            pb_ratio: 市净率Series
        
        Returns:
            因子值
        """
        return -pb_ratio.replace([np.inf, -np.inf], np.nan).fillna(pb_ratio.median())
    
    @staticmethod
    def dividend_yield_factor(div_yield):
        """
        股息率因子(越高越好)
        
        Args:
            div_yield: 股息率Series
        
        Returns:
            因子值
        """
        return div_yield.fillna(0)
    
    @staticmethod
    def ev_ebitda_factor(ev, ebitda):
        """
        EV/EBITDA因子(越低越好)
        
        Args:
            ev: 企业价值
            ebitda: 息税折旧摊销前利润
        
        Returns:
            因子值
        """
        ratio = ev / ebitda
        return -ratio.replace([np.inf, -np.inf], np.nan).fillna(ratio.median())


class GrowthFactor:
    """成长因子"""
    
    @staticmethod
    def revenue_growth_factor(revenue_growth):
        """
        营收增长率因子
        
        Args:
            revenue_growth: 营收增长率Series
        
        Returns:
            因子值
        """
        return revenue_growth.fillna(0)
    
    @staticmethod
    def profit_growth_factor(profit_growth):
        """
        净利润增长率因子
        
        Args:
            profit_growth: 净利润增长率Series
        
        Returns:
            因子值
        """
        return profit_growth.fillna(0)
    
    @staticmethod
    def roe_growth_factor(roe_current, roe_previous):
        """
        ROE提升因子
        
        Args:
            roe_current: 当期ROE
            roe_previous: 上期ROE
        
        Returns:
            因子值
        """
        return (roe_current - roe_previous).fillna(0)


class QualityFactor:
    """质量因子"""
    
    @staticmethod
    def roe_factor(roe):
        """
        ROE因子(越高越好)
        
        Args:
            roe: 净资产收益率Series
        
        Returns:
            因子值
        """
        return roe.fillna(roe.median())
    
    @staticmethod
    def roa_factor(roa):
        """
        ROA因子(越高越好)
        
        Args:
            roa: 总资产收益率Series
        
        Returns:
            因子值
        """
        return roa.fillna(roa.median())
    
    @staticmethod
    def debt_ratio_factor(debt_ratio):
        """
        资产负债率因子(越低越好)
        
        Args:
            debt_ratio: 资产负债率Series
        
        Returns:
            因子值
        """
        return -debt_ratio.fillna(debt_ratio.median())
    
    @staticmethod
    def current_ratio_factor(current_ratio):
        """
        流动比率因子(越高越好)
        
        Args:
            current_ratio: 流动比率Series
        
        Returns:
            因子值
        """
        return current_ratio.fillna(current_ratio.median())
    
    @staticmethod
    def gross_margin_factor(gross_margin):
        """
        毛利率因子(越高越好)
        
        Args:
            gross_margin: 毛利率Series
        
        Returns:
            因子值
        """
        return gross_margin.fillna(gross_margin.median())
    
    @staticmethod
    def operating_cash_flow_factor(ocf, revenue):
        """
        经营现金流/营收因子
        
        Args:
            ocf: 经营现金流
            revenue: 营业收入
        
        Returns:
            因子值
        """
        ratio = ocf / revenue
        return ratio.fillna(ratio.median())


class MomentumFactor:
    """动量因子"""
    
    @staticmethod
    def price_momentum(close_prices, lookback_days=120):
        """
        价格动量因子
        
        Args:
            close_prices: 收盘价DataFrame(index为日期, columns为股票代码)
            lookback_days: 回看天数(默认120个交易日约6个月)
        
        Returns:
            因子值Series
        """
        if len(close_prices) < lookback_days:
            return pd.Series()
        
        # 计算收益率
        momentum = close_prices.iloc[-1] / close_prices.iloc[-lookback_days] - 1
        
        return momentum
    
    @staticmethod
    def price_momentum_3m(close_prices):
        """3个月动量"""
        return MomentumFactor.price_momentum(close_prices, 60)
    
    @staticmethod
    def price_momentum_6m(close_prices):
        """6个月动量"""
        return MomentumFactor.price_momentum(close_prices, 120)
    
    @staticmethod
    def price_momentum_12m(close_prices):
        """12个月动量"""
        return MomentumFactor.price_momentum(close_prices, 240)
    
    @staticmethod
    def residual_momentum(close_prices, benchmark_prices, lookback_days=120):
        """
        残差动量(剔除市场因素)
        
        Args:
            close_prices: 股票收盘价DataFrame
            benchmark_prices: 基准价格Series
            lookback_days: 回看天数
        
        Returns:
            因子值Series
        """
        if len(close_prices) < lookback_days or len(benchmark_prices) < lookback_days:
            return pd.Series()
        
        # 计算收益率
        stock_returns = close_prices.pct_change().iloc[-lookback_days:]
        benchmark_returns = benchmark_prices.pct_change().iloc[-lookback_days:]
        
        # 计算残差动量
        residual_momentum = {}
        
        for stock in close_prices.columns:
            # 回归计算alpha
            y = stock_returns[stock].values
            x = benchmark_returns.values
            
            # 简单线性回归
            x_mean = x.mean()
            y_mean = y.mean()
            
            beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            alpha = y_mean - beta * x_mean
            
            # 残差动量 = alpha
            residual_momentum[stock] = alpha * lookback_days
        
        return pd.Series(residual_momentum)


class MultiFactorStrategy:
    """多因子选股策略"""
    
    def __init__(self, factor_weights=None):
        """
        初始化多因子策略
        
        Args:
            factor_weights: 因子权重字典,例如:
                {
                    'value': 0.3,
                    'growth': 0.25,
                    'quality': 0.25,
                    'momentum': 0.2
                }
        """
        self.factor_weights = factor_weights or {
            'value': 0.25,
            'growth': 0.25,
            'quality': 0.25,
            'momentum': 0.25
        }
    
    def calculate_composite_score(self, stock_data):
        """
        计算综合得分
        
        Args:
            stock_data: 股票数据字典,包含各类因子数据
                {
                    'pe': Series,
                    'pb': Series,
                    'roe': Series,
                    'revenue_growth': Series,
                    ...
                }
        
        Returns:
            综合得分Series
        """
        model = FactorModel()
        
        # 价值因子
        if 'pe' in stock_data:
            model.add_factor('pe', ValueFactor.pe_factor(stock_data['pe']), 
                           self.factor_weights.get('value', 0.25) * 0.4)
        if 'pb' in stock_data:
            model.add_factor('pb', ValueFactor.pb_factor(stock_data['pb']), 
                           self.factor_weights.get('value', 0.25) * 0.3)
        if 'div_yield' in stock_data:
            model.add_factor('div_yield', ValueFactor.dividend_yield_factor(stock_data['div_yield']), 
                           self.factor_weights.get('value', 0.25) * 0.3)
        
        # 成长因子
        if 'revenue_growth' in stock_data:
            model.add_factor('revenue_growth', GrowthFactor.revenue_growth_factor(stock_data['revenue_growth']), 
                           self.factor_weights.get('growth', 0.25) * 0.5)
        if 'profit_growth' in stock_data:
            model.add_factor('profit_growth', GrowthFactor.profit_growth_factor(stock_data['profit_growth']), 
                           self.factor_weights.get('growth', 0.25) * 0.5)
        
        # 质量因子
        if 'roe' in stock_data:
            model.add_factor('roe', QualityFactor.roe_factor(stock_data['roe']), 
                           self.factor_weights.get('quality', 0.25) * 0.5)
        if 'debt_ratio' in stock_data:
            model.add_factor('debt_ratio', QualityFactor.debt_ratio_factor(stock_data['debt_ratio']), 
                           self.factor_weights.get('quality', 0.25) * 0.3)
        if 'gross_margin' in stock_data:
            model.add_factor('gross_margin', QualityFactor.gross_margin_factor(stock_data['gross_margin']), 
                           self.factor_weights.get('quality', 0.25) * 0.2)
        
        # 动量因子
        if 'momentum_6m' in stock_data:
            model.add_factor('momentum_6m', stock_data['momentum_6m'], 
                           self.factor_weights.get('momentum', 0.25))
        
        return model.calculate_scores()
    
    def select_stocks(self, stock_data, n=10):
        """
        选股
        
        Args:
            stock_data: 股票数据字典
            n: 选股数量
        
        Returns:
            选中的股票代码列表及得分
        """
        scores = self.calculate_composite_score(stock_data)
        
        if scores.empty:
            return []
        
        # 选择得分最高的N只股票
        selected = scores.nlargest(n)
        
        return selected


def simple_multi_factor_backtest(stock_pool, stock_data, price_data, 
                                rebalance_freq=60, top_n=10):
    """
    简化的多因子回测
    
    Args:
        stock_pool: 股票池列表
        stock_data: 股票因子数据字典
        price_data: 价格数据DataFrame
        rebalance_freq: 调仓频率(交易日)
        top_n: 持仓数量
    
    Returns:
        回测结果DataFrame
    """
    strategy = MultiFactorStrategy()
    
    results = []
    holdings = []
    
    for i in range(rebalance_freq, len(price_data), rebalance_freq):
        # 计算当期因子得分
        scores = strategy.calculate_composite_score(stock_data)
        
        if scores.empty:
            continue
        
        # 选择股票
        selected = scores.nlargest(top_n).index.tolist()
        
        # 计算收益
        if i + rebalance_freq < len(price_data):
            period_return = price_data[selected].iloc[i:i+rebalance_freq].pct_change().mean().mean()
            
            results.append({
                'date': price_data.index[i],
                'return': period_return,
                'holdings': selected
            })
    
    return pd.DataFrame(results)


def get_current_factor_signal(stock_data):
    """
    获取当前因子信号(用于main.py集成)
    
    Args:
        stock_data: 单只股票的因子数据字典
    
    Returns:
        信号字典
    """
    strategy = MultiFactorStrategy()
    
    # 计算综合得分
    score = strategy.calculate_composite_score(stock_data)
    
    if score.empty:
        return {
            'signal': '数据不足',
            'confidence': 0,
            'score': 0
        }
    
    # 根据得分判断信号
    score_value = score.iloc[0] if isinstance(score, pd.Series) else score
    
    if score_value > 1.0:
        signal = '强烈买入'
        confidence = 80
    elif score_value > 0.5:
        signal = '买入'
        confidence = 70
    elif score_value > 0:
        signal = '偏买'
        confidence = 60
    elif score_value > -0.5:
        signal = '偏卖'
        confidence = 60
    else:
        signal = '卖出'
        confidence = 70
    
    return {
        'signal': signal,
        'confidence': confidence,
        'score': score_value
    }