# 量化交易策略系统 - 使用指南

## 目录
1. [快速开始](#快速开始)
2. [基础使用](#基础使用)
3. [高级功能](#高级功能)
4. [实战案例](#实战案例)

---

## 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd trade

# 安装依赖
pip install -r requirements.txt

# 或者使用 pip 安装
pip install -e .
```

### 2. 基础使用 - 运行策略回测

```bash
# 最简单的使用方式
python main.py sh.600519  # 分析贵州茅台

# 指定初始资金
python main.py sh.600519 200000  # 20万资金

# 指定用户持仓(1000股,成本价1800元)
python main.py sh.600519 200000 1000 1800.00
```

### 3. 查看结果

运行后会输出:
- ✅ 各策略的回测收益
- ✅ 买卖信号建议
- ✅ 持仓盈亏分析
- ✅ 止盈止损建议

---

## 基础使用

### 一、技术指标策略

#### 1.1 单策略使用

```python
# 示例: 使用MACD策略
from data_manager import load_stock_data
from macd_strategy import get_current_signal

# 加载数据
df = load_stock_data('sh.600519')

# 获取当前信号
signal = get_current_signal(df)
print(f"信号: {signal['signal']}")
print(f"置信度: {signal['confidence']}%")
```

#### 1.2 运行完整回测

```python
from data_manager import load_stock_data
from macd_strategy import macd_strategy

# 加载数据
df = load_stock_data('sh.600519', start_date='2020-01-01')

# 运行策略
trades, equity_curve = macd_strategy(df, initial_capital=100000)

print(f"总收益率: {(equity_curve.iloc[-1]/100000-1)*100:.2f}%")
```

### 二、风险管理

#### 2.1 仓位管理

```python
from risk_manager import PositionManager, calculate_position_size

# 方法1: 使用简化函数
shares = calculate_position_size(
    capital=100000,           # 总资金10万
    entry_price=50.0,         # 入场价50元
    stop_loss_price=45.0,     # 止损价45元
    risk_pct=0.02             # 单笔风险2%
)
print(f"建议买入: {shares} 股")

# 方法2: 使用凯利公式
pm = PositionManager(total_capital=100000)
kelly_pct = pm.kelly_position(
    win_rate=0.55,            # 胜率55%
    win_loss_ratio=2.0        # 盈亏比2:1
)
print(f"凯利仓位: {kelly_pct:.2%}")

# 方法3: 风险平价(多资产)
assets_vol = {
    'stock1': 0.25,  # 波动率25%
    'stock2': 0.30,  # 波动率30%
    'stock3': 0.20   # 波动率20%
}
weights = pm.risk_parity(assets_vol)
print(f"资产配置权重: {weights}")
```

#### 2.2 止损策略

```python
from risk_manager import get_stop_loss_price, StopLossStrategy
from data_manager import load_stock_data

# 加载数据
df = load_stock_data('sh.600519')

# 方法1: 固定止损
stop_price = get_stop_loss_price(
    entry_price=50.0,
    method='fixed',
    stop_pct=0.08  # 8%止损
)
print(f"止损价: {stop_price:.2f} 元")

# 方法2: ATR止损
stop_price = get_stop_loss_price(
    entry_price=50.0,
    df=df,
    method='atr',
    atr_mult=2.0
)
print(f"ATR止损价: {stop_price:.2f} 元")

# 方法3: 移动止损
stop_price = get_stop_loss_price(
    entry_price=50.0,
    method='trailing',
    highest_price=60.0,  # 最高价
    trail_pct=0.05       # 回撤5%
)
print(f"移动止损价: {stop_price:.2f} 元")
```

#### 2.3 风险监控

```python
from risk_manager import RiskMonitor
import pandas as pd

# 初始化监控器
monitor = RiskMonitor(max_drawdown=0.20)  # 最大回撤20%

# 模拟收益序列
returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])

# 计算VaR
var_95 = monitor.calculate_var(returns, confidence=0.95)
print(f"95% VaR: {var_95:.2%}")

# 生成风险报告
report = monitor.risk_report(returns, portfolio_value=100000)
print(report)
```

### 三、回测分析

#### 3.1 计算性能指标

```python
from backtest_analytics import PerformanceMetrics, calculate_all_metrics
import pandas as pd

# 创建资金曲线
equity_curve = pd.Series([100000, 105000, 103000, 108000, 110000])

# 计算各项指标
metrics = PerformanceMetrics()

sharpe = metrics.sharpe_ratio(equity_curve)
max_dd = metrics.max_drawdown(equity_curve)
total_return = metrics.total_return(equity_curve)

print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd:.2%}")
print(f"总收益率: {total_return:.2%}")

# 一键计算所有指标
all_metrics = calculate_all_metrics(equity_curve)
for key, value in all_metrics.items():
    print(f"{key}: {value}")
```

#### 3.2 交易成本分析

```python
from backtest_analytics import TransactionCost

# 初始化成本模型
tc = TransactionCost(
    commission_rate=0.0003,  # 万三佣金
    stamp_duty=0.001,        # 千一印花税
    slippage_rate=0.001      # 千一滑点
)

# 计算买入成本
buy_cost = tc.calculate_buy_cost(price=50.0, shares=1000)
print(f"买入成本: {buy_cost:.2f} 元")

# 计算卖出收入
sell_revenue = tc.calculate_sell_cost(price=55.0, shares=1000)
print(f"卖出收入: {sell_revenue:.2f} 元")

# 计算总交易成本
total_cost = tc.total_transaction_cost(
    buy_price=50.0,
    sell_price=55.0,
    shares=1000
)
print(f"总交易成本: {total_cost:.2f} 元")
```

#### 3.3 生成回测报告

```python
from backtest_analytics import BacktestReport
import pandas as pd

# 创建资金曲线和交易列表
equity_curve = pd.Series([100000, 105000, 103000, 108000, 110000])
trades = [
    (500, '2024-01-10', 'buy'),
    (-300, '2024-01-20', 'sell'),
    (800, '2024-02-01', 'buy'),
]

# 生成报告
report = BacktestReport(equity_curve, trades)
report.print_report()

# 获取月度收益
monthly_returns = report.get_monthly_returns()
print(monthly_returns)
```

---

## 高级功能

### 四、多因子选股

#### 4.1 基础使用

```python
from multi_factor import MultiFactorStrategy, ValueFactor, QualityFactor
import pandas as pd

# 准备股票数据(示例)
stock_data = {
    'pe': pd.Series({'600519': 30, '000858': 25, '600036': 8}),
    'pb': pd.Series({'600519': 8, '000858': 6, '600036': 1.2}),
    'roe': pd.Series({'600519': 0.28, '000858': 0.25, '600036': 0.15}),
    'revenue_growth': pd.Series({'600519': 0.15, '000858': 0.12, '600036': 0.08})
}

# 创建策略
strategy = MultiFactorStrategy(
    factor_weights={
        'value': 0.3,
        'growth': 0.3,
        'quality': 0.2,
        'momentum': 0.2
    }
)

# 计算综合得分
scores = strategy.calculate_composite_score(stock_data)
print("各股票得分:")
print(scores.sort_values(ascending=False))

# 选股
selected = strategy.select_stocks(stock_data, n=2)
print(f"\n推荐股票: {selected}")
```

#### 4.2 单因子分析

```python
from multi_factor import ValueFactor

# PE因子分析
pe_series = pd.Series({'600519': 30, '000858': 25, '600036': 8})
pe_factor = ValueFactor.pe_factor(pe_series)
print("PE因子值(越低越好):")
print(pe_factor)

# ROE因子分析
from multi_factor import QualityFactor
roe_series = pd.Series({'600519': 0.28, '000858': 0.25, '600036': 0.15})
roe_factor = QualityFactor.roe_factor(roe_series)
print("\nROE因子值:")
print(roe_factor)
```

### 五、动量策略

#### 5.1 价格动量

```python
from momentum_strategy import PriceMomentum, get_current_momentum_signal
from data_manager import load_stock_data

# 加载数据
df = load_stock_data('sh.600519')

# 方法1: 简化函数
signal = get_current_momentum_signal(df['close'], lookback=120)
print(f"动量信号: {signal['signal']}")
print(f"动量值: {signal['momentum']:.2%}")

# 方法2: 详细分析
strategy = PriceMomentum(lookback_period=120, holding_period=20)

# 计算风险调整动量
risk_adj_momentum = strategy.calculate_risk_adjusted_momentum(df['close'])
print(f"风险调整动量: {risk_adj_momentum:.2f}")
```

#### 5.2 横截面动量(多股票)

```python
from momentum_strategy import CrossSectionalMomentum
import pandas as pd

# 准备多股票数据(示例)
price_data = pd.DataFrame({
    '600519': [1800, 1850, 1820, 1900, 1880],
    '000858': [200, 210, 205, 220, 215],
    '600036': [30, 32, 31, 35, 34]
})

# 创建策略
strategy = CrossSectionalMomentum(
    lookback=4,
    top_pct=0.3,      # 买前30%
    bottom_pct=0.3    # 卖后30%
)

# 选择组合
long_portfolio, short_portfolio = strategy.select_portfolio(price_data)
print(f"多头组合: {long_portfolio}")
print(f"空头组合: {short_portfolio}")
```

### 六、配对交易

#### 6.1 寻找协整配对

```python
from pair_trading import find_tradable_pairs
import pandas as pd

# 准备多股票价格数据
price_data = pd.DataFrame({
    '600519': [...],  # 贵州茅台价格序列
    '000858': [...],  # 五粮液价格序列
    '600036': [...]   # 招商银行价格序列
})

# 寻找可交易配对
pairs = find_tradable_pairs(
    price_data,
    min_correlation=0.7,   # 最小相关系数
    max_pvalue=0.05        # 最大p值
)

print("发现的协整配对:")
for pair in pairs:
    print(f"{pair['stock1']} - {pair['stock2']}")
    print(f"  相关系数: {pair['correlation']:.2f}")
    print(f"  p值: {pair['pvalue']:.4f}")
```

#### 6.2 配对交易信号

```python
from pair_trading import get_current_pair_signal
from data_manager import load_stock_data

# 加载两只股票数据
df1 = load_stock_data('sh.600519')  # 贵州茅台
df2 = load_stock_data('sz.000858')  # 五粮液

# 获取配对交易信号
signal = get_current_pair_signal(
    df1['close'],
    df2['close'],
    window=20,
    entry_threshold=2.0
)

print(f"信号: {signal['signal']}")
print(f"Z-score: {signal['z_score']:.2f}")
print(f"对冲比例: {signal['hedge_ratio']:.2f}")
print(f"置信度: {signal['confidence']}%")
```

### 七、参数优化

#### 7.1 网格搜索

```python
from parameter_optimizer import GridSearch
from ma_crossover import ma_crossover_strategy

# 定义参数网格
param_grid = {
    'short_period': [5, 10, 15],
    'long_period': [20, 30, 60]
}

# 创建优化器
optimizer = GridSearch(
    strategy_func=ma_crossover_strategy,
    param_grid=param_grid,
    metric='sharpe'  # 优化目标:夏普比率
)

# 运行优化
best_params, best_score, results = optimizer.run(data)

print(f"最优参数: {best_params}")
print(f"最优夏普比率: {best_score:.2f}")

# 获取前5个最优参数
top_5 = optimizer.get_top_n_params(5)
for i, result in enumerate(top_5, 1):
    print(f"\n第{i}名: {result['params']}")
    print(f"  夏普比率: {result['score']:.2f}")
```

#### 7.2 遗传算法优化

```python
from parameter_optimizer import GeneticAlgorithm
from ma_crossover import ma_crossover_strategy

# 定义参数边界
param_bounds = {
    'short_period': (5, 30),    # 短期均线范围
    'long_period': (20, 100)    # 长期均线范围
}

# 创建优化器
optimizer = GeneticAlgorithm(
    strategy_func=ma_crossover_strategy,
    param_bounds=param_bounds,
    population_size=50,   # 种群大小
    generations=20,       # 迭代代数
    mutation_rate=0.1,    # 变异率
    metric='sharpe'
)

# 运行优化
best_params, best_score = optimizer.run(data)
print(f"最优参数: {best_params}")
print(f"最优夏普比率: {best_score:.2f}")
```

#### 7.3 滚动前向分析

```python
from parameter_optimizer import WalkForwardAnalysis, GridSearch

# 创建优化器
optimizer = GridSearch(strategy_func, param_grid, metric='sharpe')

# 创建滚动前向分析
wfa = WalkForwardAnalysis(
    strategy_func=strategy_func,
    optimizer=optimizer,
    train_period=240,  # 训练期240天
    test_period=60     # 测试期60天
)

# 运行分析
results = wfa.run(data)

print("滚动前向测试结果:")
for result in results:
    print(f"\n训练期: {result['train_start']} 至 {result['train_end']}")
    print(f"测试期: {result['test_start']} 至 {result['test_end']}")
    print(f"最优参数: {result['best_params']}")
```

---

## 实战案例

### 案例1: 完整的股票分析流程

```python
from data_manager import load_stock_data
from risk_manager import calculate_position_size, get_stop_loss_price
from backtest_analytics import calculate_all_metrics
from macd_strategy import macd_strategy
from momentum_strategy import get_current_momentum_signal

# 1. 加载数据
symbol = 'sh.600519'
df = load_stock_data(symbol)
print(f"加载 {len(df)} 天数据")

# 2. 运行策略回测
trades, equity_curve = macd_strategy(df, initial_capital=100000)

# 3. 计算性能指标
metrics = calculate_all_metrics(equity_curve, trades)
print("\n=== 回测结果 ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

# 4. 获取当前信号
from macd_strategy import get_current_signal
signal = get_current_signal(df)
print(f"\n当前信号: {signal['signal']} (置信度: {signal['confidence']}%)")

# 5. 动量分析
momentum = get_current_momentum_signal(df['close'])
print(f"动量: {momentum['signal']} (动量值: {momentum['momentum']:.2%})")

# 6. 风险管理建议
current_price = df['close'].iloc[-1]
stop_loss = get_stop_loss_price(current_price, df, method='atr')
position = calculate_position_size(
    capital=100000,
    entry_price=current_price,
    stop_loss_price=stop_loss
)

print(f"\n=== 风险管理建议 ===")
print(f"当前价格: {current_price:.2f} 元")
print(f"建议止损价: {stop_loss:.2f} 元")
print(f"建议持仓: {position} 股")
print(f"建议仓位: {position * current_price / 100000:.2%}")
```

### 案例2: 多股票组合管理

```python
from data_manager import load_stock_data
from risk_manager import PositionManager
from multi_factor import MultiFactorStrategy
import pandas as pd

# 1. 准备股票池
stock_pool = ['sh.600519', 'sz.000858', 'sh.600036']

# 2. 收集数据
stock_data = {}
price_data = {}

for stock in stock_pool:
    df = load_stock_data(stock)
    # 这里简化,实际需要获取PE、PB等基本面数据
    price_data[stock] = df['close']

# 3. 多因子选股
strategy = MultiFactorStrategy()
# scores = strategy.calculate_composite_score(stock_data)
# selected = strategy.select_stocks(stock_data, n=2)

# 4. 风险平价配置
pm = PositionManager(total_capital=100000)

# 计算各股票波动率
volatility = {}
for stock, prices in price_data.items():
    returns = prices.pct_change()
    volatility[stock] = returns.std() * (252 ** 0.5)  # 年化波动率

# 风险平价权重
weights = pm.risk_parity(volatility)
print("\n=== 风险平价配置 ===")
for stock, weight in weights.items():
    print(f"{stock}: {weight:.2%}")
```

### 案例3: 策略参数优化实战

```python
from data_manager import load_stock_data
from parameter_optimizer import optimize_strategy_params, cross_validate_strategy
from ma_crossover import ma_crossover_strategy

# 1. 加载数据
data = load_stock_data('sh.600519')

# 2. 参数网格
param_grid = {
    'short_period': [5, 10, 15, 20],
    'long_period': [30, 40, 50, 60]
}

# 3. 网格搜索
print("=== 网格搜索优化 ===")
best_params, best_score, _ = optimize_strategy_params(
    strategy_func=ma_crossover_strategy,
    data=data,
    param_grid=param_grid,
    method='grid',
    metric='sharpe',
    verbose=True
)

# 4. 交叉验证
print("\n=== 5折交叉验证 ===")
cv_results = cross_validate_strategy(
    strategy_func=ma_crossover_strategy,
    data=data,
    params=best_params,
    n_splits=5
)

# 5. 使用最优参数回测
trades, equity = ma_crossover_strategy(data, **best_params)
print(f"\n最终回测收益: {(equity.iloc[-1]/100000-1)*100:.2f}%")
```

---

## 常见问题

### Q1: 如何添加新策略?

1. 创建新策略文件(如 `my_strategy.py`)
2. 实现 `get_current_signal(df)` 函数
3. 在 `main.py` 中导入并调用

### Q2: 如何获取基本面数据?

目前系统主要使用技术指标,基本面数据需要:
- 从财经API获取(Tushare、AKShare等)
- 或手动整理CSV文件

### Q3: 如何调整风险参数?

在调用风险管理函数时传入参数:
```python
# 调整止损比例
stop_loss = get_stop_loss_price(price, method='fixed', stop_pct=0.10)

# 调整单笔风险
position = calculate_position_size(capital, price, stop_loss, risk_pct=0.03)
```

### Q4: 回测结果与实际交易差异大?

可能原因:
- 未考虑交易成本 → 使用 `TransactionCost` 模型
- 参数过拟合 → 使用滚动前向分析验证
- 未来函数 → 检查策略逻辑

---

## 下一步

1. ✅ 运行基础示例,熟悉各模块
2. ✅ 尝试参数优化,找到最优参数
3. ✅ 组合多个策略,构建交易系统
4. ✅ 添加风控规则,实盘模拟

**重要提示**: 本系统仅供学习和研究使用,不构成投资建议。实盘交易请谨慎评估风险!