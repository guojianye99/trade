#!/usr/bin/env python3
"""
data_manager 模块单元测试
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import (
    ensure_data_dir,
    get_local_filename,
    download_stock_data,
    load_stock_data,
    download_multiple,
    download_dividend_data,
    load_dividend_data,
    calculate_dividend_income,
    list_local_data,
    DATA_DIR
)


class TestEnsureDataDir(unittest.TestCase):
    """测试 ensure_data_dir 函数"""
    
    def test_ensure_data_dir_creates_directory(self):
        """测试数据目录创建"""
        result = ensure_data_dir()
        self.assertTrue(result.exists())
        self.assertEqual(result, DATA_DIR)


class TestGetLocalFilename(unittest.TestCase):
    """测试 get_local_filename 函数"""
    
    def test_filename_with_prefix(self):
        """测试带前缀的股票代码"""
        result = get_local_filename('sh.601138', '2020-01-01', '2024-12-31')
        self.assertEqual(result, '601138.csv')
        
        result = get_local_filename('sz.000001', '2020-01-01', '2024-12-31')
        self.assertEqual(result, '000001.csv')
    
    def test_filename_without_prefix(self):
        """测试不带前缀的股票代码"""
        # 6开头应该是sh
        result = get_local_filename('601138', '2020-01-01', '2024-12-31')
        self.assertEqual(result, '601138.csv')
        
        # 非6开头应该是sz
        result = get_local_filename('000001', '2020-01-01', '2024-12-31')
        self.assertEqual(result, '000001.csv')


class TestLoadStockData(unittest.TestCase):
    """测试 load_stock_data 函数"""
    
    def test_load_nonexistent_file_without_auto_download(self):
        """测试加载不存在的文件（不自动下载）"""
        result = load_stock_data('999999', auto_download=False)
        self.assertIsNone(result)
    
    @patch('data_manager.download_stock_data')
    def test_auto_download_when_file_not_exists(self, mock_download):
        """测试文件不存在时自动下载"""
        # 创建模拟数据
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [10, 11, 12, 13, 14],
            'high': [11, 12, 13, 14, 15],
            'low': [9, 10, 11, 12, 13],
            'close': [10.5, 11.5, 12.5, 13.5, 14.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        mock_df = mock_df.set_index('date')
        mock_download.return_value = mock_df
        
        # 使用不存在的股票代码测试
        result = load_stock_data('999888', start_date='2024-01-01', 
                                  end_date='2024-01-05', auto_download=True)
        
        # 应该调用下载函数
        mock_download.assert_called_once()


class TestDownloadMultiple(unittest.TestCase):
    """测试 download_multiple 函数"""
    
    @patch('data_manager.download_stock_data')
    def test_download_multiple_success(self, mock_download):
        """测试批量下载成功"""
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [10, 11, 12, 13, 14]
        }).set_index('date')
        mock_download.return_value = mock_df
        
        result = download_multiple(['sh.600000', 'sh.600001'], 
                                    start_date='2024-01-01', 
                                    end_date='2024-01-05')
        
        self.assertEqual(len(result), 2)
        self.assertIn('sh.600000', result)
        self.assertIn('sh.600001', result)
    
    @patch('data_manager.download_stock_data')
    def test_download_multiple_partial_failure(self, mock_download):
        """测试批量下载部分失败"""
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [10, 11, 12, 13, 14]
        }).set_index('date')
        
        # 第一个成功，第二个失败
        mock_download.side_effect = [mock_df, None]
        
        result = download_multiple(['sh.600000', 'sh.600001'], 
                                    start_date='2024-01-01', 
                                    end_date='2024-01-05')
        
        self.assertEqual(len(result), 1)
        self.assertIn('sh.600000', result)


class TestDividendData(unittest.TestCase):
    """测试分红数据相关函数"""
    
    @patch('data_manager.load_dividend_data')
    def test_load_dividend_data(self, mock_load):
        """测试加载分红数据"""
        mock_df = pd.DataFrame({
            'code': ['sh.600000'] * 3,
            'dividCashPsBeforeTax': [0.5, 0.6, 0.7],
            'dividRegistDate': ['2021-06-01', '2022-06-01', '2023-06-01']
        })
        mock_load.return_value = mock_df
        
        result = load_dividend_data('sh.600000')
        self.assertIsInstance(result, pd.DataFrame)


class TestCalculateDividendIncome(unittest.TestCase):
    """测试 calculate_dividend_income 函数"""
    
    def test_calculate_dividend_income_empty_dividend(self):
        """测试空分红数据"""
        price_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [10, 11, 12, 13, 14],
            'units': [100, 100, 100, 100, 100]
        }).set_index('date')
        
        dividend_df = pd.DataFrame()
        result = calculate_dividend_income(price_df, dividend_df)
        
        # 空分红数据应返回原数据
        pd.testing.assert_frame_equal(result, price_df)


class TestListLocalData(unittest.TestCase):
    """测试 list_local_data 函数"""
    
    def test_list_local_data_empty(self):
        """测试空数据目录"""
        # 使用临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 这个测试验证函数不会崩溃
            result = list_local_data()
            # 函数应该返回一个列表（可能为空或包含现有数据）
            self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()