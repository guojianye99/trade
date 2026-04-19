"""
回测分析模块 - 提供回测性能指标、交易成本模型、基准对比等功能
"""
import numpy as np
import pandas as pd
from datetime import datetime


class TransactionCost:
    """交易成本模型"""
    
    def __init__(self, commission_rate=0.0003, stamp_duty=0.001, 
                 min_commission=5, slippage_rate=0.001):
        """
        初始化交易成本模型
        
        Args:
            commission_rate: 佣金费率(默认万分之三)
            stamp_duty: 印花税率(默认千分之一,仅卖出)
            min_commission: 最低佣金(默认5元)
            slippage_rate: 滑点率(默认千分之一)
        """
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.min_commission = min_commission
        self.slippage_rate = slippage_rate
    
    def calculate_buy_cost(self, price, shares):
        """
        计算买入成本
        
        Args:
            price: 买入价格
            shares: 买入股数
        
        Returns:
            总成本(包含佣金和滑点)
        """
        # 滑点调整价格
        actual_price = price * (1 + self.slippage_rate)
        
        # 佣金
        commission = max(actual_price * shares * self.commission_rate, self.min_commission)
        
        total_cost = actual_price * shares + commission
        
        return total_cost
    
    def calculate_sell_cost(self, price, shares):
        """
        计算卖出成本
        
        Args:
            price: 卖出价格
            shares: 卖出股数
        
        Returns:
            实际收入(扣除佣金、印花税和滑点)
        """
        # 滑点调整价格
        actual_price = price * (1 - self.slippage_rate)
        
        # 佣金
        commission = max(actual_price * shares * self.commission_rate, self.min_commission)
        
        # 印花税(仅卖出)
        stamp = actual_price * shares * self.stamp_duty
        
        total_cost = commission + stamp
        
        actual_revenue = actual_price * shares - total_cost
        
        return actual_revenue
    
    def total_transaction_cost(self, buy_price, sell_price, shares):
        """
        计算完整交易的成本
        
        Args:
            buy_price: 买入价格
            sell_price: 卖出价格
            shares: 交易股数
        
        Returns:
            总交易成本
        """
        buy_cost = self.calculate_buy_cost(buy_price, shares)
        sell_revenue = self.calculate_sell_cost(sell_price, shares)
        
        # 名义收益
        nominal_profit = (sell_price - buy_price) * shares
        
        # 实际收益
        actual_profit = sell_revenue - buy_cost
        
        # 交易成本
        total_cost = nominal_profit - actual_profit
        
        return total_cost


class PerformanceMetrics:
    """性能指标计算"""
    
    @staticmethod
    def calculate_returns(equity_curve):
        """
        计算收益率序列
        
        Args:
            equity_curve: 资金曲线
        
        Returns:
            收益率序列
        """
        return equity_curve.pct_change().dropna()
    
    @staticmethod
    def total_return(equity_curve):
        """
        计算总收益率
        
        Args:
            equity_curve: 资金曲线
        
        Returns:
            总收益率
        """
        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    @staticmethod
    def annualized_return(equity_curve, trading_days=252):
        """
        计算年化收益率
        
        Args:
            equity_curve: 资金曲线
            trading_days: 年交易日数(默认252)
        
        Returns:
            年化收益率
        """
        total_ret = PerformanceMetrics.total_return(equity_curve)
        days = len(equity_curve)
        
        if days == 0:
            return 0
        
        annualized = (1 + total_ret) ** (trading_days / days) - 1
        
        return annualized
    
    @staticmethod
    def max_drawdown(equity_curve):
        """
        计算最大回撤
        
        Args:
            equity_curve: 资金曲线
        
        Returns:
            最大回撤比例
        """
        # 计算累计最大值
        running_max = equity_curve.cummax()
        
        # 计算回撤
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def sharpe_ratio(equity_curve, risk_free_rate=0.03, trading_days=252):
        """
        计算夏普比率
        
        Args:
            equity_curve: 资金曲线
            risk_free_rate: 无风险利率(年化,默认3%)
            trading_days: 年交易日数
        
        Returns:
            夏普比率
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # 年化无风险利率转换为日利率
        daily_rf = risk_free_rate / trading_days
        
        # 超额收益
        excess_returns = returns - daily_rf
        
        # 夏普比率
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(trading_days)
        
        return sharpe
    
    @staticmethod
    def sortino_ratio(equity_curve, risk_free_rate=0.03, trading_days=252):
        """
        计算索提诺比率(仅考虑下行风险)
        
        Args:
            equity_curve: 资金曲线
            risk_free_rate: 无风险利率
            trading_days: 年交易日数
        
        Returns:
            索提诺比率
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        if len(returns) == 0:
            return 0
        
        # 年化无风险利率转换为日利率
        daily_rf = risk_free_rate / trading_days
        
        # 超额收益
        excess_returns = returns - daily_rf
        
        # 下行波动率(仅负收益)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_std = negative_returns.std() * np.sqrt(trading_days)
        
        # 索提诺比率
        sortino = excess_returns.mean() * trading_days / downside_std
        
        return sortino
    
    @staticmethod
    def win_rate(trades):
        """
        计算胜率
        
        Args:
            trades: 交易列表,每个元素为(profit, ...)
        
        Returns:
            胜率
        """
        if len(trades) == 0:
            return 0
        
        wins = sum(1 for trade in trades if trade[0] > 0)
        
        return wins / len(trades)
    
    @staticmethod
    def profit_loss_ratio(trades):
        """
        计算盈亏比
        
        Args:
            trades: 交易列表,每个元素为(profit, ...)
        
        Returns:
            盈亏比
        """
        profits = [trade[0] for trade in trades if trade[0] > 0]
        losses = [abs(trade[0]) for trade in trades if trade[0] < 0]
        
        if len(losses) == 0:
            return float('inf') if len(profits) > 0 else 0
        
        avg_profit = np.mean(profits) if len(profits) > 0 else 0
        avg_loss = np.mean(losses)
        
        return avg_profit / avg_loss if avg_loss > 0 else 0
    
    @staticmethod
    def calmar_ratio(equity_curve, trading_days=252):
        """
        计算卡玛比率(年化收益/最大回撤)
        
        Args:
            equity_curve: 资金曲线
            trading_days: 年交易日数
        
        Returns:
            卡玛比率
        """
        annualized_ret = PerformanceMetrics.annualized_return(equity_curve, trading_days)
        max_dd = abs(PerformanceMetrics.max_drawdown(equity_curve))
        
        if max_dd == 0:
            return float('inf') if annualized_ret > 0 else 0
        
        return annualized_ret / max_dd
    
    @staticmethod
    def information_ratio(equity_curve, benchmark_curve, trading_days=252):
        """
        计算信息比率
        
        Args:
            equity_curve: 策略资金曲线
            benchmark_curve: 基准资金曲线
            trading_days: 年交易日数
        
        Returns:
            信息比率
        """
        # 确保长度一致
        min_len = min(len(equity_curve), len(benchmark_curve))
        equity_curve = equity_curve.iloc[:min_len]
        benchmark_curve = benchmark_curve.iloc[:min_len]
        
        # 计算收益率
        strategy_returns = PerformanceMetrics.calculate_returns(equity_curve)
        benchmark_returns = PerformanceMetrics.calculate_returns(benchmark_curve)
        
        # 超额收益
        excess_returns = strategy_returns - benchmark_returns
        
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0
        
        # 信息比率
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
        
        return ir


class BacktestReport:
    """回测报告生成器"""
    
    def __init__(self, equity_curve, trades=None, benchmark_curve=None):
        """
        初始化回测报告
        
        Args:
            equity_curve: 资金曲线(Series, index为日期)
            trades: 交易列表
            benchmark_curve: 基准资金曲线
        """
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.benchmark_curve = benchmark_curve
        self.metrics = PerformanceMetrics()
    
    def generate_report(self):
        """
        生成完整的回测报告
        
        Returns:
            报告字典
        """
        # 基本指标
        total_ret = self.metrics.total_return(self.equity_curve)
        annualized_ret = self.metrics.annualized_return(self.equity_curve)
        max_dd = self.metrics.max_drawdown(self.equity_curve)
        sharpe = self.metrics.sharpe_ratio(self.equity_curve)
        sortino = self.metrics.sortino_ratio(self.equity_curve)
        calmar = self.metrics.calmar_ratio(self.equity_curve)
        
        # 交易指标
        win_rate = self.metrics.win_rate(self.trades)
        pl_ratio = self.metrics.profit_loss_ratio(self.trades)
        
        report = {
            '总收益率': f"{total_ret:.2%}",
            '年化收益率': f"{annualized_ret:.2%}",
            '最大回撤': f"{max_dd:.2%}",
            '夏普比率': f"{sharpe:.2f}",
            '索提诺比率': f"{sortino:.2f}",
            '卡玛比率': f"{calmar:.2f}",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{pl_ratio:.2f}",
            '总交易次数': len(self.trades),
            '初始资金': f"{self.equity_curve.iloc[0]:.2f}",
            '最终资金': f"{self.equity_curve.iloc[-1]:.2f}",
            '回测天数': len(self.equity_curve)
        }
        
        # 如果有基准,计算相对指标
        if self.benchmark_curve is not None:
            ir = self.metrics.information_ratio(self.equity_curve, self.benchmark_curve)
            benchmark_ret = self.metrics.total_return(self.benchmark_curve)
            
            report['基准收益率'] = f"{benchmark_ret:.2%}"
            report['信息比率'] = f"{ir:.2f}"
            report['超额收益'] = f"{total_ret - benchmark_ret:.2%}"
        
        return report
    
    def print_report(self):
        """打印回测报告"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print(" "*15 + "回测分析报告")
        print("="*60)
        
        for key, value in report.items():
            print(f"{key:12s}: {value}")
        
        print("="*60 + "\n")
    
    def get_monthly_returns(self):
        """
        计算月度收益
        
        Returns:
            月度收益DataFrame
        """
        returns = self.metrics.calculate_returns(self.equity_curve)
        
        # 按月分组
        monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        monthly_df = monthly.reset_index()
        monthly_df.columns = ['年', '月', '收益率']
        
        return monthly_df
    
    def get_yearly_returns(self):
        """
        计算年度收益
        
        Returns:
            年度收益DataFrame
        """
        returns = self.metrics.calculate_returns(self.equity_curve)
        
        # 按年分组
        yearly = returns.groupby(returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        yearly_df = yearly.reset_index()
        yearly_df.columns = ['年', '收益率']
        
        return yearly_df


def calculate_all_metrics(equity_curve, trades=None, benchmark_curve=None):
    """
    计算所有回测指标的简化函数
    
    Args:
        equity_curve: 资金曲线
        trades: 交易列表
        benchmark_curve: 基准曲线
    
    Returns:
        指标字典
    """
    report = BacktestReport(equity_curve, trades, benchmark_curve)
    return report.generate_report()


def apply_transaction_cost(df, initial_capital, commission_rate=0.0003, 
                          stamp_duty=0.001, slippage_rate=0.001):
    """
    应用交易成本到回测结果
    
    Args:
        df: 包含position和close列的DataFrame
        initial_capital: 初始资金
        commission_rate: 佣金费率
        stamp_duty: 印花税率
        slippage_rate: 滑点率
    
    Returns:
        包含交易成本的资金曲线
    """
    tc = TransactionCost(commission_rate, stamp_duty, slippage_rate=slippage_rate)
    
    capital = initial_capital
    shares = 0
    equity_curve = []
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        # 买入
        if df['position'].iloc[i] == 'buy' and shares == 0:
            cost = tc.calculate_buy_cost(current_price, int(capital / current_price))
            shares = int((capital - cost + current_price * int(capital / current_price)) / current_price)
            capital = capital - cost
        
        # 卖出
        elif df['position'].iloc[i] == 'sell' and shares > 0:
            revenue = tc.calculate_sell_cost(current_price, shares)
            capital = capital + revenue
            shares = 0
        
        # 记录权益
        equity = capital + shares * current_price
        equity_curve.append(equity)
    
    return pd.Series(equity_curve, index=df.index)