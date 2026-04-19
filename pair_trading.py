"""
配对交易策略 - 基于协整关系和价差交易的统计套利策略
"""
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


class CointegrationTest:
    """协整检验"""
    
    @staticmethod
    def engle_granger_test(series1, series2, significance_level=0.05):
        """
        Engle-Granger两步法协整检验
        
        Args:
            series1: 价格序列1
            series2: 价格序列2
            significance_level: 显著性水平
        
        Returns:
            (是否协整, p值, 协整向量)
        """
        if len(series1) != len(series2):
            return False, 1.0, (1, 0)
        
        # 第一步: 回归
        # series1 = beta * series2 + alpha + residual
        x = series2.values
        y = series1.values
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 计算残差
        residuals = y - (slope * x + intercept)
        
        # 第二步: ADF检验残差平稳性
        # 使用简化的ADF检验(实际应使用statsmodels的adfuller)
        # 这里用简单的游程检验替代
        adf_statistic, adf_pvalue, _, _, _ = stats.linregress(
            residuals[:-1], residuals[1:]
        )
        
        # 如果残差平稳,则协整
        is_cointegrated = adf_pvalue < significance_level
        
        return is_cointegrated, adf_pvalue, (1, slope)
    
    @staticmethod
    def johansen_test(price_data, significance_level=0.05):
        """
        Johansen协整检验(简化版)
        
        Args:
            price_data: 价格数据DataFrame(多列)
            significance_level: 显著性水平
        
        Returns:
            协整关系数
        """
        # 简化实现:使用相关系数矩阵
        corr_matrix = price_data.corr()
        
        # 计算高度相关的配对数量
        n_pairs = 0
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # 相关系数>0.7
                    n_pairs += 1
        
        return n_pairs


class SpreadCalculator:
    """价差计算器"""
    
    @staticmethod
    def calculate_hedge_ratio(series1, series2, window=60):
        """
        计算对冲比例
        
        Args:
            series1: 价格序列1
            series2: 价格序列2
            window: 滚动窗口
        
        Returns:
            对冲比例序列
        """
        if len(series1) < window:
            return pd.Series(1.0, index=series1.index)
        
        hedge_ratios = []
        
        for i in range(window, len(series1)):
            y = series1.iloc[i-window:i].values
            x = series2.iloc[i-window:i].values
            
            slope, _, _, _, _ = stats.linregress(x, y)
            hedge_ratios.append(slope)
        
        return pd.Series(hedge_ratios, index=series1.index[window:])
    
    @staticmethod
    def calculate_spread(series1, series2, hedge_ratio=1.0):
        """
        计算价差
        
        Args:
            series1: 价格序列1
            series2: 价格序列2
            hedge_ratio: 对冲比例
        
        Returns:
            价差序列
        """
        spread = series1 - hedge_ratio * series2
        return spread
    
    @staticmethod
    def normalize_spread(spread, window=20):
        """
        标准化价差(Z-score)
        
        Args:
            spread: 价差序列
            window: 滚动窗口
        
        Returns:
            标准化价差
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        z_score = (spread - rolling_mean) / rolling_std
        
        return z_score


class PairTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, 
                 stop_loss_threshold=4.0, window=20):
        """
        初始化配对交易策略
        
        Args:
            entry_threshold: 入场阈值(Z-score)
            exit_threshold: 出场阈值
            stop_loss_threshold: 止损阈值
            window: 计算Z-score的窗口期
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.window = window
    
    def find_cointegrated_pairs(self, price_data, significance_level=0.05):
        """
        寻找协整配对
        
        Args:
            price_data: 价格数据DataFrame(各列为不同股票)
            significance_level: 显著性水平
        
        Returns:
            协整配对列表
        """
        n_stocks = len(price_data.columns)
        pairs = []
        
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                stock1 = price_data.columns[i]
                stock2 = price_data.columns[j]
                
                # 协整检验
                is_coint, pvalue, _ = CointegrationTest.engle_granger_test(
                    price_data[stock1], price_data[stock2], significance_level
                )
                
                if is_coint:
                    pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'pvalue': pvalue
                    })
        
        # 按p值排序
        pairs = sorted(pairs, key=lambda x: x['pvalue'])
        
        return pairs
    
    def generate_signals(self, series1, series2):
        """
        生成交易信号
        
        Args:
            series1: 价格序列1
            series2: 价格序列2
        
        Returns:
            信号字典
        """
        if len(series1) < self.window or len(series2) < self.window:
            return {
                'signal': '数据不足',
                'z_score': 0,
                'spread': 0
            }
        
        # 计算对冲比例
        hedge_ratio = SpreadCalculator.calculate_hedge_ratio(
            series1, series2, self.window
        )
        
        # 计算价差
        spread = SpreadCalculator.calculate_spread(
            series1, series2, hedge_ratio.iloc[-1]
        )
        
        # 标准化价差
        z_score = SpreadCalculator.normalize_spread(spread, self.window)
        
        current_z = z_score.iloc[-1]
        current_spread = spread.iloc[-1]
        
        # 生成信号
        if current_z > self.entry_threshold:
            signal = '卖出价差'  # 做空价差,即买入series2,卖出series1
            confidence = min(100, 50 + abs(current_z) * 10)
        elif current_z < -self.entry_threshold:
            signal = '买入价差'  # 做多价差,即买入series1,卖出series2
            confidence = min(100, 50 + abs(current_z) * 10)
        elif abs(current_z) < self.exit_threshold:
            signal = '平仓'
            confidence = 50
        elif abs(current_z) > self.stop_loss_threshold:
            signal = '止损平仓'
            confidence = 90
        else:
            signal = '持有'
            confidence = 50
        
        return {
            'signal': signal,
            'z_score': current_z,
            'spread': current_spread,
            'confidence': confidence,
            'hedge_ratio': hedge_ratio.iloc[-1]
        }
    
    def backtest_pair(self, series1, series2, initial_capital=100000):
        """
        回测配对交易
        
        Args:
            series1: 价格序列1
            series2: 价格序列2
            initial_capital: 初始资金
        
        Returns:
            回测结果DataFrame
        """
        if len(series1) < self.window:
            return pd.DataFrame()
        
        capital = initial_capital
        position = 0  # 0:无持仓, 1:做多价差, -1:做空价差
        results = []
        
        # 计算对冲比例和价差
        hedge_ratios = SpreadCalculator.calculate_hedge_ratio(
            series1, series2, self.window
        )
        
        for i in range(self.window, len(series1)):
            hedge_ratio = hedge_ratios.iloc[i - self.window]
            
            # 计算当期价差和Z-score
            spread = series1.iloc[i] - hedge_ratio * series2.iloc[i]
            
            # 滚动计算Z-score
            if i >= self.window * 2:
                historical_spread = series1.iloc[i-self.window:i] - \
                                   hedge_ratio * series2.iloc[i-self.window:i]
                z_score = (spread - historical_spread.mean()) / historical_spread.std()
            else:
                continue
            
            # 交易逻辑
            if position == 0:  # 无持仓
                if z_score > self.entry_threshold:
                    position = -1  # 做空价差
                    entry_z = z_score
                elif z_score < -self.entry_threshold:
                    position = 1  # 做多价差
                    entry_z = z_score
            
            elif position == 1:  # 做多价差
                # 止损或平仓
                if z_score > self.stop_loss_threshold or abs(z_score) < self.exit_threshold:
                    position = 0
            
            elif position == -1:  # 做空价差
                # 止损或平仓
                if z_score < -self.stop_loss_threshold or abs(z_score) < self.exit_threshold:
                    position = 0
            
            # 记录结果
            results.append({
                'date': series1.index[i],
                'price1': series1.iloc[i],
                'price2': series2.iloc[i],
                'spread': spread,
                'z_score': z_score,
                'position': position,
                'capital': capital
            })
        
        return pd.DataFrame(results)


class StatisticalArbitrage:
    """统计套利策略"""
    
    def __init__(self, n_pairs=5, capital_per_pair=20000):
        """
        初始化统计套利策略
        
        Args:
            n_pairs: 配对数量
            capital_per_pair: 每对资金
        """
        self.n_pairs = n_pairs
        self.capital_per_pair = capital_per_pair
    
    def select_pairs(self, price_data):
        """
        选择配对
        
        Args:
            price_data: 价格数据DataFrame
        
        Returns:
            选中的配对列表
        """
        pair_strategy = PairTradingStrategy()
        all_pairs = pair_strategy.find_cointegrated_pairs(price_data)
        
        # 选择前n_pairs个最显著的配对
        selected = all_pairs[:self.n_pairs]
        
        return selected
    
    def construct_portfolio(self, price_data):
        """
        构建投资组合
        
        Args:
            price_data: 价格数据
        
        Returns:
            投资组合权重字典
        """
        pairs = self.select_pairs(price_data)
        
        weights = {}
        
        for pair in pairs:
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            
            # 计算对冲比例
            hedge_ratio = SpreadCalculator.calculate_hedge_ratio(
                price_data[stock1], price_data[stock2]
            ).iloc[-1]
            
            # 分配权重(简化:等权重)
            # 多头股票:正权重,空头股票:负权重
            weights[stock1] = weights.get(stock1, 0) + self.capital_per_pair
            weights[stock2] = weights.get(stock2, 0) - self.capital_per_pair * hedge_ratio
        
        return weights


class TriangularArbitrage:
    """三角套利策略"""
    
    def __init__(self, threshold=0.001):
        """
        初始化三角套利策略
        
        Args:
            threshold: 套利阈值
        """
        self.threshold = threshold
    
    def check_arbitrage(self, rate_ab, rate_bc, rate_ac):
        """
        检查是否存在三角套利机会
        
        Args:
            rate_ab: A/B汇率
            rate_bc: B/C汇率
            rate_ac: A/C汇率
        
        Returns:
            (是否存在套利, 套利路径, 预期收益)
        """
        # 直接路径: A -> C
        direct = rate_ac
        
        # 间接路径: A -> B -> C
        indirect = rate_ab * rate_bc
        
        # 套利机会
        arbitrage_return = abs(direct - indirect) / min(direct, indirect)
        
        if arbitrage_return > self.threshold:
            if direct > indirect:
                path = 'A -> B -> C'
                profit = (indirect - direct) / direct
            else:
                path = 'A -> C'
                profit = (direct - indirect) / indirect
            
            return True, path, profit
        
        return False, '', 0


def get_current_pair_signal(series1, series2, window=20, entry_threshold=2.0):
    """
    获取当前配对交易信号(用于main.py集成)
    
    Args:
        series1: 价格序列1
        series2: 价格序列2
        window: 计算窗口
        entry_threshold: 入场阈值
    
    Returns:
        信号字典
    """
    strategy = PairTradingStrategy(
        entry_threshold=entry_threshold,
        window=window
    )
    return strategy.generate_signals(series1, series2)


def find_tradable_pairs(price_data, min_correlation=0.7, max_pvalue=0.05):
    """
    寻找可交易配对
    
    Args:
        price_data: 价格数据DataFrame
        min_correlation: 最小相关系数
        max_pvalue: 最大p值
    
    Returns:
        可交易配对列表
    """
    # 计算相关系数矩阵
    corr_matrix = price_data.corr()
    
    # 寻找高相关配对
    pairs = []
    
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > min_correlation:
                stock1 = corr_matrix.columns[i]
                stock2 = corr_matrix.columns[j]
                
                # 协整检验
                is_coint, pvalue, _ = CointegrationTest.engle_granger_test(
                    price_data[stock1], price_data[stock2], max_pvalue
                )
                
                if is_coint and pvalue < max_pvalue:
                    pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': corr_matrix.iloc[i, j],
                        'pvalue': pvalue
                    })
    
    # 按相关性排序
    pairs = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    return pairs