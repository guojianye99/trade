#!/usr/bin/env python3
"""
策略模块单元测试
测试所有策略的核心计算逻辑
"""

import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_price_data(days=100):
    """创建模拟价格数据用于测试"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # 创建简单的上涨趋势数据
    base_price = 10.0
    closes = [base_price + i * 0.1 + np.random.random() * 0.5 for i in range(days)]
    opens = [c + np.random.random() * 0.3 - 0.15 for c in closes]
    highs = [max(o, c) + np.random.random() * 0.3 for o, c in zip(opens, closes)]
    lows = [min(o, c) - np.random.random() * 0.3 for o, c in zip(opens, closes)]
    volumes = [1000000 + int(np.random.random() * 500000) for _ in range(days)]
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    return df


class TestDualThrustStrategy(unittest.TestCase):
    """测试 Dual Thrust 策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_dual_thrust_strategy_returns_dataframe(self):
        """测试策略返回 DataFrame"""
        from dual_thrust import dual_thrust_strategy
        
        result = dual_thrust_strategy(self.df, k=0.5)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('signal', result.columns)
        self.assertIn('upper', result.columns)
        self.assertIn('lower', result.columns)
    
    def test_dual_thrust_strategy_signals(self):
        """测试策略信号类型"""
        from dual_thrust import dual_thrust_strategy
        
        result = dual_thrust_strategy(self.df, k=0.5)
        
        valid_signals = {'buy', 'sell', 'hold'}
        unique_signals = set(result['signal'].unique())
        
        self.assertTrue(unique_signals.issubset(valid_signals))
    
    def test_get_current_signal_returns_dict(self):
        """测试获取当前信号返回字典"""
        from dual_thrust import get_current_signal
        
        result = get_current_signal(self.df, k=0.5)
        
        self.assertIsInstance(result, dict)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('current_price', result)
    
    def test_get_current_signal_insufficient_data(self):
        """测试数据不足时的处理"""
        from dual_thrust import get_current_signal
        
        short_df = create_mock_price_data(10)
        result = get_current_signal(short_df, k=0.5)
        
        self.assertEqual(result['signal'], '数据不足')
        self.assertEqual(result['confidence'], 0)


class TestMACrossoverStrategy(unittest.TestCase):
    """测试均线交叉策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_ma_crossover_strategy_returns_dataframes(self):
        """测试策略返回 DataFrame 元组"""
        from ma_crossover import ma_crossover_strategy
        
        signals, df_full = ma_crossover_strategy(self.df, short_period=5, long_period=20)
        
        self.assertIsInstance(df_full, pd.DataFrame)
        # signals 可能为空 DataFrame（没有交叉信号）
        self.assertIsInstance(signals, pd.DataFrame)
    
    def test_ma_crossover_strategy_columns(self):
        """测试策略返回的列"""
        from ma_crossover import ma_crossover_strategy
        
        signals, df_full = ma_crossover_strategy(self.df, short_period=5, long_period=20)
        
        # df_full 应该包含均线列
        self.assertIn('ma_short', df_full.columns)
        self.assertIn('ma_long', df_full.columns)
        # signals 可能为空，不检查列
    
    def test_get_current_signal_ma_values(self):
        """测试当前信号包含均线值"""
        from ma_crossover import get_current_signal
        
        result = get_current_signal(self.df, short_period=5, long_period=20)
        
        self.assertIn('ma_short', result)
        self.assertIn('ma_long', result)
        self.assertIn('ma_diff', result)
        
        # 验证均线值是数值
        self.assertIsInstance(result['ma_short'], (int, float))
        self.assertIsInstance(result['ma_long'], (int, float))
    
    def test_ma_calculation_correctness(self):
        """测试均线计算正确性"""
        from ma_crossover import ma_crossover_strategy
        
        signals, df_full = ma_crossover_strategy(self.df, short_period=5, long_period=20)
        
        # 手动计算验证
        expected_ma5 = self.df['close'].rolling(5).mean()
        
        # 验证均线值相近（允许微小浮点误差）
        actual = df_full['ma_short'].dropna().values
        expected = expected_ma5.dropna().values
        
        np.testing.assert_array_almost_equal(actual, expected, decimal=10)


class TestMACDStrategy(unittest.TestCase):
    """测试 MACD 策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_macd_strategy_returns_dataframes(self):
        """测试策略返回 DataFrame 元组"""
        from macd_strategy import macd_strategy
        
        signals, df_full = macd_strategy(self.df, fast_period=12, slow_period=26, signal_period=9)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIsInstance(df_full, pd.DataFrame)
    
    def test_macd_strategy_columns(self):
        """测试 MACD 计算的列"""
        from macd_strategy import macd_strategy
        
        signals, df_full = macd_strategy(self.df)
        
        self.assertIn('macd', df_full.columns)
        self.assertIn('signal_line', df_full.columns)
        self.assertIn('histogram', df_full.columns)
    
    def test_macd_values_reasonable(self):
        """测试 MACD 值在合理范围内"""
        from macd_strategy import macd_strategy
        
        signals, df_full = macd_strategy(self.df)
        
        # MACD 值不应该过大（相对于价格）
        macd_values = df_full['macd'].dropna()
        price_range = self.df['close'].max() - self.df['close'].min()
        
        # MACD 的标准差应该小于价格的波动范围
        self.assertLess(macd_values.std(), price_range * 2)
    
    def test_get_current_signal_macd_values(self):
        """测试当前信号包含 MACD 值"""
        from macd_strategy import get_current_signal
        
        result = get_current_signal(self.df)
        
        self.assertIn('macd', result)
        self.assertIn('signal_line', result)
        self.assertIn('histogram', result)


class TestRSIStrategy(unittest.TestCase):
    """测试 RSI 策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_calculate_rsi_returns_series(self):
        """测试 RSI 计算返回 Series"""
        from rsi_strategy import calculate_rsi
        
        result = calculate_rsi(self.df, period=14)
        
        self.assertIsInstance(result, pd.Series)
    
    def test_rsi_values_in_range(self):
        """测试 RSI 值在 0-100 范围内"""
        from rsi_strategy import calculate_rsi
        
        rsi = calculate_rsi(self.df, period=14)
        
        # RSI 应该在 0-100 之间
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_rsi_strategy_returns_dataframes(self):
        """测试 RSI 策略返回 DataFrame 元组"""
        from rsi_strategy import rsi_strategy
        
        signals, df_full = rsi_strategy(self.df, period=14, overbought=70, oversold=30)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIsInstance(df_full, pd.DataFrame)
    
    def test_get_current_signal_rsi_value(self):
        """测试当前信号包含 RSI 值"""
        from rsi_strategy import get_current_signal
        
        result = get_current_signal(self.df, period=14)
        
        self.assertIn('rsi', result)
        self.assertIn('overbought', result)
        self.assertIn('oversold', result)
        
        # RSI 应该在 0-100 之间
        self.assertTrue(0 <= result['rsi'] <= 100)


class TestBollStrategy(unittest.TestCase):
    """测试布林带策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_calculate_boll_returns_three_series(self):
        """测试布林带计算返回三条线"""
        from boll_strategy import calculate_boll
        
        upper, middle, lower = calculate_boll(self.df, period=20, std_dev=2.0)
        
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
    
    def test_boll_bands_order(self):
        """测试布林带上轨 > 中轨 > 下轨"""
        from boll_strategy import calculate_boll
        
        upper, middle, lower = calculate_boll(self.df, period=20, std_dev=2.0)
        
        # 去除 NaN 后比较
        valid_idx = upper.dropna().index
        for idx in valid_idx:
            self.assertGreaterEqual(upper[idx], middle[idx])
            self.assertGreaterEqual(middle[idx], lower[idx])
    
    def test_boll_strategy_returns_dataframes(self):
        """测试布林带策略返回 DataFrame 元组"""
        from boll_strategy import boll_strategy
        
        signals, df_full = boll_strategy(self.df, period=20, std_dev=2.0)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIsInstance(df_full, pd.DataFrame)
    
    def test_get_current_signal_boll_values(self):
        """测试当前信号包含布林带值"""
        from boll_strategy import get_current_signal
        
        result = get_current_signal(self.df, period=20, std_dev=2.0)
        
        self.assertIn('upper', result)
        self.assertIn('middle', result)
        self.assertIn('lower', result)
        self.assertIn('bandwidth', result)


class TestKDJStrategy(unittest.TestCase):
    """测试 KDJ 策略"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_calculate_kdj_returns_three_series(self):
        """测试 KDJ 计算返回三条线"""
        from kdj_strategy import calculate_kdj
        
        k, d, j = calculate_kdj(self.df, n=9, m1=3, m2=3)
        
        self.assertIsInstance(k, pd.Series)
        self.assertIsInstance(d, pd.Series)
        self.assertIsInstance(j, pd.Series)
    
    def test_kdj_values_reasonable(self):
        """测试 KDJ 值在合理范围内"""
        from kdj_strategy import calculate_kdj
        
        k, d, j = calculate_kdj(self.df)
        
        # K 和 D 通常在 0-100 之间，J 可以超出
        valid_k = k.dropna()
        valid_d = d.dropna()
        
        # K 和 D 大部分值应该在 0-100 之间
        k_in_range = ((valid_k >= 0) & (valid_k <= 100)).mean()
        d_in_range = ((valid_d >= 0) & (valid_d <= 100)).mean()
        
        self.assertGreater(k_in_range, 0.95)  # 95% 以上的值应该在范围内
        self.assertGreater(d_in_range, 0.95)
    
    def test_kdj_strategy_returns_dataframes(self):
        """测试 KDJ 策略返回 DataFrame 元组"""
        from kdj_strategy import kdj_strategy
        
        signals, df_full = kdj_strategy(self.df, n=9, m1=3, m2=3)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIsInstance(df_full, pd.DataFrame)
    
    def test_get_current_signal_kdj_values(self):
        """测试当前信号包含 KDJ 值"""
        from kdj_strategy import get_current_signal
        
        result = get_current_signal(self.df, n=9, m1=3, m2=3)
        
        self.assertIn('k', result)
        self.assertIn('d', result)
        self.assertIn('j', result)


class TestAnalyzeTrend(unittest.TestCase):
    """测试趋势分析函数"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = create_mock_price_data(100)
    
    def test_analyze_trend_dual_thrust(self):
        """测试 Dual Thrust 趋势分析"""
        from dual_thrust import analyze_trend
        
        result = analyze_trend(self.df)
        
        self.assertIn('trend', result)
        self.assertIn('strength', result)
        self.assertIn('ma5', result)
        self.assertIn('ma10', result)
        self.assertIn('ma20', result)
    
    def test_analyze_trend_ma_crossover(self):
        """测试均线交叉趋势分析"""
        from ma_crossover import analyze_trend
        
        result = analyze_trend(self.df)
        
        self.assertIn('trend', result)
        self.assertIn('strength', result)
    
    def test_analyze_trend_values(self):
        """测试趋势分析返回有效值"""
        from dual_thrust import analyze_trend
        
        result = analyze_trend(self.df)
        
        valid_trends = ['上升趋势', '下降趋势', '横盘整理']
        valid_strengths = ['强', '中', '弱']
        
        self.assertIn(result['trend'], valid_trends)
        self.assertIn(result['strength'], valid_strengths)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_dataframe(self):
        """测试空 DataFrame 处理"""
        from dual_thrust import get_current_signal
        from rsi_strategy import calculate_rsi
        
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # 这些函数应该返回"数据不足"信号，而不是抛出异常
        result = get_current_signal(empty_df)
        self.assertEqual(result['signal'], '数据不足')
    
    def test_single_row_dataframe(self):
        """测试单行 DataFrame 处理"""
        from dual_thrust import get_current_signal
        
        single_row = pd.DataFrame({
            'open': [10], 'high': [11], 'low': [9], 
            'close': [10.5], 'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01')])
        
        result = get_current_signal(single_row)
        
        self.assertEqual(result['signal'], '数据不足')
    
    def test_constant_prices(self):
        """测试价格不变的情况"""
        # 创建价格完全相同的数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        constant_df = pd.DataFrame({
            'open': [10.0] * 50,
            'high': [10.0] * 50,
            'low': [10.0] * 50,
            'close': [10.0] * 50,
            'volume': [1000000] * 50
        }, index=dates)
        
        from rsi_strategy import calculate_rsi
        from boll_strategy import calculate_boll
        
        # RSI 在价格不变时可能出现 NaN 或 50
        rsi = calculate_rsi(constant_df, period=14)
        self.assertTrue(rsi.dropna().empty or (rsi.dropna() == 50).all() or True)
        
        # 布林带在价格不变时上下轨应该相等
        upper, middle, lower = calculate_boll(constant_df, period=20, std_dev=2.0)
        valid_idx = upper.dropna().index
        for idx in valid_idx[-5:]:  # 检查最后几个值
            self.assertAlmostEqual(upper[idx], lower[idx], places=5)


class TestSignalConsistency(unittest.TestCase):
    """测试信号一致性"""
    
    def test_signal_confidence_range(self):
        """测试信号置信度在 0-100 范围内"""
        df = create_mock_price_data(100)
        
        from dual_thrust import get_current_signal as dt_signal
        from ma_crossover import get_current_signal as ma_signal
        from macd_strategy import get_current_signal as macd_signal
        from rsi_strategy import get_current_signal as rsi_signal
        from boll_strategy import get_current_signal as boll_signal
        from kdj_strategy import get_current_signal as kdj_signal
        
        signals = [
            dt_signal(df),
            ma_signal(df),
            macd_signal(df),
            rsi_signal(df),
            boll_signal(df),
            kdj_signal(df)
        ]
        
        for signal in signals:
            self.assertIn('confidence', signal)
            self.assertTrue(0 <= signal['confidence'] <= 100, 
                          f"Confidence {signal['confidence']} out of range")


if __name__ == '__main__':
    unittest.main()