"""
风险管理模块 - 提供仓位管理、止损策略、风险监控等功能
"""
import numpy as np
import pandas as pd
from datetime import datetime


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, total_capital, max_position_pct=0.95, risk_per_trade=0.02):
        """
        初始化仓位管理器
        
        Args:
            total_capital: 总资金
            max_position_pct: 最大仓位比例(默认95%)
            risk_per_trade: 单笔交易风险比例(默认2%)
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
    
    def kelly_position(self, win_rate, win_loss_ratio, max_kelly=0.25):
        """
        凯利公式计算最优仓位
        
        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比(平均盈利/平均亏损)
            max_kelly: 最大凯利比例(默认25%,防止过度杠杆)
        
        Returns:
            建议仓位比例
        """
        # 凯利公式: f = (p * b - q) / b
        # p = 胜率, q = 1-p, b = 盈亏比
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # 限制最大仓位
        kelly = max(0, min(kelly, max_kelly))
        
        return kelly
    
    def fixed_fractional(self, entry_price, stop_loss_price):
        """
        固定分数法计算仓位
        
        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格
        
        Returns:
            建议持仓股数
        """
        if stop_loss_price >= entry_price:
            return 0
        
        # 单笔最大亏损金额
        max_loss_amount = self.total_capital * self.risk_per_trade
        
        # 单股最大亏损
        loss_per_share = entry_price - stop_loss_price
        
        # 计算股数
        shares = int(max_loss_amount / loss_per_share)
        
        # 限制最大仓位
        max_shares = int(self.total_capital * self.max_position_pct / entry_price)
        
        return min(shares, max_shares)
    
    def volatility_adjusted(self, current_price, atr, target_volatility=0.02):
        """
        波动率调整仓位
        
        Args:
            current_price: 当前价格
            atr: 平均真实波幅
            target_volatility: 目标波动率(默认2%)
        
        Returns:
            建议仓位比例
        """
        if atr <= 0:
            return 0
        
        # 基于ATR的仓位比例
        position_pct = (target_volatility * current_price) / atr
        
        # 限制最大仓位
        position_pct = min(position_pct, self.max_position_pct)
        
        return position_pct
    
    def risk_parity(self, assets_volatility):
        """
        风险平价模型 - 使每个资产的风险贡献相等
        
        Args:
            assets_volatility: 各资产的波动率字典 {asset: volatility}
        
        Returns:
            各资产的权重字典 {asset: weight}
        """
        if not assets_volatility:
            return {}
        
        # 计算风险倒数
        inv_vol = {asset: 1/vol for asset, vol in assets_volatility.items()}
        
        # 归一化得到权重
        total_inv_vol = sum(inv_vol.values())
        weights = {asset: inv_v/total_inv_vol for asset, inv_v in inv_vol.items()}
        
        return weights


class StopLossStrategy:
    """止损策略"""
    
    @staticmethod
    def fixed_stop_loss(entry_price, stop_loss_pct=0.08):
        """
        固定百分比止损
        
        Args:
            entry_price: 入场价格
            stop_loss_pct: 止损比例(默认8%)
        
        Returns:
            止损价格
        """
        return entry_price * (1 - stop_loss_pct)
    
    @staticmethod
    def atr_stop_loss(df, entry_price, atr_multiplier=2.0, lookback=14):
        """
        ATR止损
        
        Args:
            df: 价格数据DataFrame
            entry_price: 入场价格
            atr_multiplier: ATR倍数(默认2倍)
            lookback: ATR计算周期(默认14)
        
        Returns:
            止损价格
        """
        if len(df) < lookback:
            return entry_price * 0.92
        
        # 计算ATR
        high = df['high'].iloc[-lookback:]
        low = df['low'].iloc[-lookback:]
        close = df['close'].iloc[-lookback-1:-1]
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()
        
        stop_loss = entry_price - atr_multiplier * atr
        
        return max(stop_loss, entry_price * 0.8)  # 最大止损20%
    
    @staticmethod
    def trailing_stop(current_price, highest_price, trail_pct=0.05):
        """
        移动止损
        
        Args:
            current_price: 当前价格
            highest_price: 持仓期间最高价
            trail_pct: 回撤比例(默认5%)
        
        Returns:
            止损价格
        """
        return highest_price * (1 - trail_pct)
    
    @staticmethod
    def sar_stop_loss(df, af=0.02, max_af=0.2):
        """
        抛物线止损(SAR)
        
        Args:
            df: 价格数据DataFrame
            af: 加速因子
            max_af: 最大加速因子
        
        Returns:
            SAR止损价格
        """
        if len(df) < 5:
            return df['close'].iloc[-1] * 0.92
        
        # 简化的SAR计算
        high = df['high'].iloc[-5:]
        low = df['low'].iloc[-5:]
        
        # 极值点
        ep = high.max()  # 极值点价格
        sar = low.iloc[-1]  # 初始SAR
        
        # 迭代计算
        for i in range(len(high)):
            sar = sar + af * (ep - sar)
            sar = min(sar, low.iloc[i])
        
        return sar


class RiskMonitor:
    """风险监控器"""
    
    def __init__(self, max_drawdown=0.20, max_daily_loss=0.05):
        """
        初始化风险监控器
        
        Args:
            max_drawdown: 最大回撤限制(默认20%)
            max_daily_loss: 单日最大亏损(默认5%)
        """
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.peak_value = 0
        self.current_value = 0
    
    def update(self, portfolio_value):
        """
        更新投资组合价值
        
        Args:
            portfolio_value: 当前投资组合价值
        """
        self.current_value = portfolio_value
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
    
    def calculate_drawdown(self):
        """
        计算当前回撤
        
        Returns:
            回撤比例
        """
        if self.peak_value == 0:
            return 0
        
        drawdown = (self.peak_value - self.current_value) / self.peak_value
        return drawdown
    
    def check_risk_limits(self):
        """
        检查风险限制
        
        Returns:
            (是否触发风险限制, 风险提示信息)
        """
        drawdown = self.calculate_drawdown()
        
        if drawdown >= self.max_drawdown:
            return True, f"⚠️ 触发最大回撤限制: 当前回撤 {drawdown:.2%} >= 限制 {self.max_drawdown:.2%}"
        
        if drawdown >= self.max_drawdown * 0.8:
            return False, f"⚠️ 接近最大回撤限制: 当前回撤 {drawdown:.2%}, 限制 {self.max_drawdown:.2%}"
        
        return False, "✅ 风险水平正常"
    
    def calculate_var(self, returns, confidence=0.95):
        """
        计算VaR(风险价值)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平(默认95%)
        
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0
        
        # 历史模拟法
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return var
    
    def calculate_cvar(self, returns, confidence=0.95):
        """
        计算CVaR(条件风险价值)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平(默认95%)
        
        Returns:
            CVaR值
        """
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence)
        
        # 计算小于VaR的收益的平均值
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    def risk_report(self, returns, portfolio_value):
        """
        生成风险报告
        
        Args:
            returns: 收益率序列
            portfolio_value: 当前投资组合价值
        
        Returns:
            风险报告字典
        """
        self.update(portfolio_value)
        
        drawdown = self.calculate_drawdown()
        var = self.calculate_var(returns)
        cvar = self.calculate_cvar(returns)
        is_risk, risk_msg = self.check_risk_limits()
        
        return {
            'current_value': portfolio_value,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'var_95': var,
            'cvar_95': cvar,
            'is_risk_exceeded': is_risk,
            'risk_message': risk_msg,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0  # 年化波动率
        }


def calculate_position_size(capital, entry_price, stop_loss_price, risk_pct=0.02):
    """
    简化的仓位计算函数
    
    Args:
        capital: 总资金
        entry_price: 入场价格
        stop_loss_price: 止损价格
        risk_pct: 单笔风险比例
    
    Returns:
        建议持仓股数
    """
    pm = PositionManager(capital, risk_per_trade=risk_pct)
    return pm.fixed_fractional(entry_price, stop_loss_price)


def get_stop_loss_price(entry_price, df=None, method='fixed', **kwargs):
    """
    获取止损价格
    
    Args:
        entry_price: 入场价格
        df: 价格数据(用于ATR止损)
        method: 止损方法('fixed', 'atr', 'trailing')
        **kwargs: 其他参数
    
    Returns:
        止损价格
    """
    sl = StopLossStrategy()
    
    if method == 'fixed':
        stop_pct = kwargs.get('stop_pct', 0.08)
        return sl.fixed_stop_loss(entry_price, stop_pct)
    
    elif method == 'atr' and df is not None:
        atr_mult = kwargs.get('atr_mult', 2.0)
        lookback = kwargs.get('lookback', 14)
        return sl.atr_stop_loss(df, entry_price, atr_mult, lookback)
    
    elif method == 'trailing':
        highest = kwargs.get('highest_price', entry_price)
        trail_pct = kwargs.get('trail_pct', 0.05)
        return sl.trailing_stop(entry_price, highest, trail_pct)
    
    else:
        return entry_price * 0.92  # 默认8%止损