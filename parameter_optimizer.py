"""
参数优化模块 - 提供网格搜索、贝叶斯优化、遗传算法等参数优化方法
"""
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import random


class GridSearch:
    """网格搜索优化"""
    
    def __init__(self, strategy_func, param_grid, metric='sharpe'):
        """
        初始化网格搜索
        
        Args:
            strategy_func: 策略函数,接受参数字典返回回测结果
            param_grid: 参数网格字典,例如:
                {
                    'short_period': [5, 10, 15],
                    'long_period': [20, 30, 60]
                }
            metric: 优化指标('sharpe', 'return', 'max_drawdown')
        """
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.metric = metric
        self.results = []
    
    def run(self, data, verbose=True):
        """
        运行网格搜索
        
        Args:
            data: 回测数据
            verbose: 是否打印进度
        
        Returns:
            最优参数和结果
        """
        # 生成所有参数组合
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        all_combinations = list(product(*param_values))
        
        if verbose:
            print(f"开始网格搜索,共 {len(all_combinations)} 种参数组合")
        
        best_score = -np.inf if self.metric != 'max_drawdown' else np.inf
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            try:
                # 运行策略
                result = self.strategy_func(data, **params)
                
                # 计算指标
                if self.metric == 'sharpe':
                    score = self._calculate_sharpe(result)
                elif self.metric == 'return':
                    score = self._calculate_return(result)
                elif self.metric == 'max_drawdown':
                    score = self._calculate_max_drawdown(result)
                
                # 记录结果
                self.results.append({
                    'params': params,
                    'score': score,
                    'result': result
                })
                
                # 更新最优参数
                if self.metric == 'max_drawdown':
                    if score < best_score:  # 越小越好
                        best_score = score
                        best_params = params
                else:
                    if score > best_score:  # 越大越好
                        best_score = score
                        best_params = params
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"已完成 {i+1}/{len(all_combinations)} 次迭代")
            
            except Exception as e:
                if verbose:
                    print(f"参数 {params} 运行失败: {e}")
                continue
        
        if verbose:
            print(f"\n网格搜索完成")
            print(f"最优参数: {best_params}")
            print(f"最优{self.metric}: {best_score:.4f}")
        
        return best_params, best_score, self.results
    
    def _calculate_sharpe(self, result):
        """计算夏普比率"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            returns = result['capital'].pct_change().dropna()
            if len(returns) == 0 or returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)
        return 0
    
    def _calculate_return(self, result):
        """计算总收益率"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            return (result['capital'].iloc[-1] / result['capital'].iloc[0] - 1)
        return 0
    
    def _calculate_max_drawdown(self, result):
        """计算最大回撤"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            capital = result['capital']
            running_max = capital.cummax()
            drawdown = (capital - running_max) / running_max
            return drawdown.min()
        return 0
    
    def get_top_n_params(self, n=5):
        """
        获取前N个最优参数
        
        Args:
            n: 数量
        
        Returns:
            排序后的结果列表
        """
        if self.metric == 'max_drawdown':
            sorted_results = sorted(self.results, key=lambda x: x['score'])
        else:
            sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:n]


class RandomSearch:
    """随机搜索优化"""
    
    def __init__(self, strategy_func, param_distributions, n_iter=50, metric='sharpe'):
        """
        初始化随机搜索
        
        Args:
            strategy_func: 策略函数
            param_distributions: 参数分布字典,例如:
                {
                    'short_period': [5, 10, 15, 20],
                    'long_period': range(20, 100, 10)
                }
            n_iter: 迭代次数
            metric: 优化指标
        """
        self.strategy_func = strategy_func
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.metric = metric
        self.results = []
    
    def run(self, data, verbose=True):
        """
        运行随机搜索
        
        Args:
            data: 回测数据
            verbose: 是否打印进度
        
        Returns:
            最优参数和结果
        """
        if verbose:
            print(f"开始随机搜索,共 {self.n_iter} 次迭代")
        
        best_score = -np.inf if self.metric != 'max_drawdown' else np.inf
        best_params = None
        
        for i in range(self.n_iter):
            # 随机采样参数
            params = {}
            for param_name, param_values in self.param_distributions.items():
                params[param_name] = random.choice(list(param_values))
            
            try:
                # 运行策略
                result = self.strategy_func(data, **params)
                
                # 计算指标
                if self.metric == 'sharpe':
                    score = self._calculate_sharpe(result)
                elif self.metric == 'return':
                    score = self._calculate_return(result)
                elif self.metric == 'max_drawdown':
                    score = self._calculate_max_drawdown(result)
                
                # 记录结果
                self.results.append({
                    'params': params,
                    'score': score
                })
                
                # 更新最优参数
                if self.metric == 'max_drawdown':
                    if score < best_score:
                        best_score = score
                        best_params = params
                else:
                    if score > best_score:
                        best_score = score
                        best_params = params
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"已完成 {i+1}/{self.n_iter} 次迭代")
            
            except Exception as e:
                if verbose:
                    print(f"参数 {params} 运行失败: {e}")
                continue
        
        if verbose:
            print(f"\n随机搜索完成")
            print(f"最优参数: {best_params}")
            print(f"最优{self.metric}: {best_score:.4f}")
        
        return best_params, best_score, self.results
    
    def _calculate_sharpe(self, result):
        """计算夏普比率"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            returns = result['capital'].pct_change().dropna()
            if len(returns) == 0 or returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)
        return 0
    
    def _calculate_return(self, result):
        """计算总收益率"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            return (result['capital'].iloc[-1] / result['capital'].iloc[0] - 1)
        return 0
    
    def _calculate_max_drawdown(self, result):
        """计算最大回撤"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            capital = result['capital']
            running_max = capital.cummax()
            drawdown = (capital - running_max) / running_max
            return drawdown.min()
        return 0


class GeneticAlgorithm:
    """遗传算法优化"""
    
    def __init__(self, strategy_func, param_bounds, population_size=50, 
                 generations=20, mutation_rate=0.1, metric='sharpe'):
        """
        初始化遗传算法
        
        Args:
            strategy_func: 策略函数
            param_bounds: 参数边界字典,例如:
                {
                    'short_period': (5, 30),
                    'long_period': (20, 100)
                }
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            metric: 优化指标
        """
        self.strategy_func = strategy_func
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.metric = metric
    
    def run(self, data, verbose=True):
        """
        运行遗传算法
        
        Args:
            data: 回测数据
            verbose: 是否打印进度
        
        Returns:
            最优参数和结果
        """
        if verbose:
            print(f"开始遗传算法优化,种群大小={self.population_size}, 代数={self.generations}")
        
        # 初始化种群
        population = self._initialize_population()
        
        best_individual = None
        best_score = -np.inf if self.metric != 'max_drawdown' else np.inf
        
        for gen in range(self.generations):
            # 评估适应度
            fitness_scores = []
            
            for individual in population:
                try:
                    result = self.strategy_func(data, **individual)
                    score = self._calculate_metric(result)
                    fitness_scores.append((individual, score))
                    
                    # 更新最优个体
                    if self.metric == 'max_drawdown':
                        if score < best_score:
                            best_score = score
                            best_individual = individual
                    else:
                        if score > best_score:
                            best_score = score
                            best_individual = individual
                except:
                    fitness_scores.append((individual, 0 if self.metric != 'max_drawdown' else 1))
            
            # 选择
            selected = self._selection(fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            population = self._mutation(offspring)
            
            if verbose:
                print(f"第 {gen+1}/{self.generations} 代, 最优{self.metric}: {best_score:.4f}")
        
        if verbose:
            print(f"\n遗传算法优化完成")
            print(f"最优参数: {best_individual}")
            print(f"最优{self.metric}: {best_score:.4f}")
        
        return best_individual, best_score
    
    def _initialize_population(self):
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                if isinstance(min_val, int):
                    individual[param] = random.randint(min_val, max_val)
                else:
                    individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        
        return population
    
    def _selection(self, fitness_scores):
        """选择操作(锦标赛选择)"""
        selected = []
        
        for _ in range(self.population_size):
            # 随机选择3个个体
            tournament = random.sample(fitness_scores, 3)
            
            # 选择最优的
            if self.metric == 'max_drawdown':
                winner = min(tournament, key=lambda x: x[1])
            else:
                winner = max(tournament, key=lambda x: x[1])
            
            selected.append(winner[0])
        
        return selected
    
    def _crossover(self, selected):
        """交叉操作"""
        offspring = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
            
            # 单点交叉
            child1, child2 = {}, {}
            
            for j, param in enumerate(self.param_bounds.keys()):
                if random.random() < 0.5:
                    child1[param] = parent1[param]
                    child2[param] = parent2[param]
                else:
                    child1[param] = parent2[param]
                    child2[param] = parent1[param]
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _mutation(self, offspring):
        """变异操作"""
        for individual in offspring:
            for param, (min_val, max_val) in self.param_bounds.items():
                if random.random() < self.mutation_rate:
                    if isinstance(min_val, int):
                        individual[param] = random.randint(min_val, max_val)
                    else:
                        individual[param] = random.uniform(min_val, max_val)
        
        return offspring
    
    def _calculate_metric(self, result):
        """计算优化指标"""
        if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
            if self.metric == 'sharpe':
                returns = result['capital'].pct_change().dropna()
                if len(returns) == 0 or returns.std() == 0:
                    return 0
                return returns.mean() / returns.std() * np.sqrt(252)
            elif self.metric == 'return':
                return (result['capital'].iloc[-1] / result['capital'].iloc[0] - 1)
            elif self.metric == 'max_drawdown':
                capital = result['capital']
                running_max = capital.cummax()
                drawdown = (capital - running_max) / running_max
                return drawdown.min()
        return 0


class WalkForwardAnalysis:
    """滚动前向分析"""
    
    def __init__(self, strategy_func, optimizer, train_period, test_period):
        """
        初始化滚动前向分析
        
        Args:
            strategy_func: 策略函数
            optimizer: 优化器(GridSearch或RandomSearch)
            train_period: 训练期长度(交易日)
            test_period: 测试期长度(交易日)
        """
        self.strategy_func = strategy_func
        self.optimizer = optimizer
        self.train_period = train_period
        self.test_period = test_period
    
    def run(self, data, verbose=True):
        """
        运行滚动前向分析
        
        Args:
            data: 完整数据
            verbose: 是否打印进度
        
        Returns:
            滚动前向测试结果
        """
        if len(data) < self.train_period + self.test_period:
            print("数据长度不足")
            return None
        
        results = []
        start = 0
        
        while start + self.train_period + self.test_period <= len(data):
            # 训练期数据
            train_data = data.iloc[start:start+self.train_period]
            
            # 测试期数据
            test_data = data.iloc[start+self.train_period:start+self.train_period+self.test_period]
            
            if verbose:
                print(f"\n训练期: {train_data.index[0]} 至 {train_data.index[-1]}")
                print(f"测试期: {test_data.index[0]} 至 {test_data.index[-1]}")
            
            # 在训练期优化参数
            best_params, _, _ = self.optimizer.run(train_data, verbose=False)
            
            if best_params:
                # 在测试期验证
                test_result = self.strategy_func(test_data, **best_params)
                
                results.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'best_params': best_params,
                    'test_result': test_result
                })
            
            start += self.test_period
        
        return results


def optimize_strategy_params(strategy_func, data, param_grid, method='grid', **kwargs):
    """
    策略参数优化的简化接口
    
    Args:
        strategy_func: 策略函数
        data: 回测数据
        param_grid: 参数网格
        method: 优化方法('grid', 'random', 'genetic')
        **kwargs: 其他参数
    
    Returns:
        最优参数和结果
    """
    if method == 'grid':
        optimizer = GridSearch(strategy_func, param_grid, kwargs.get('metric', 'sharpe'))
    elif method == 'random':
        optimizer = RandomSearch(
            strategy_func, param_grid, 
            kwargs.get('n_iter', 50), 
            kwargs.get('metric', 'sharpe')
        )
    elif method == 'genetic':
        optimizer = GeneticAlgorithm(
            strategy_func, param_grid,
            kwargs.get('population_size', 50),
            kwargs.get('generations', 20),
            kwargs.get('mutation_rate', 0.1),
            kwargs.get('metric', 'sharpe')
        )
    else:
        raise ValueError(f"不支持的优化方法: {method}")
    
    return optimizer.run(data, verbose=kwargs.get('verbose', True))


def cross_validate_strategy(strategy_func, data, params, n_splits=5):
    """
    交叉验证策略
    
    Args:
        strategy_func: 策略函数
        data: 完整数据
        params: 策略参数
        n_splits: 折数
    
    Returns:
        交叉验证结果
    """
    chunk_size = len(data) // n_splits
    results = []
    
    for i in range(n_splits):
        # 测试集
        test_start = i * chunk_size
        test_end = (i + 1) * chunk_size
        test_data = data.iloc[test_start:test_end]
        
        # 训练集(其余数据)
        train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])
        
        try:
            # 在测试集上运行
            result = strategy_func(test_data, **params)
            
            # 计算指标
            if isinstance(result, pd.DataFrame) and 'capital' in result.columns:
                total_return = result['capital'].iloc[-1] / result['capital'].iloc[0] - 1
                returns = result['capital'].pct_change().dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                results.append({
                    'fold': i+1,
                    'return': total_return,
                    'sharpe': sharpe
                })
        except Exception as e:
            print(f"第 {i+1} 折验证失败: {e}")
            continue
    
    # 计算平均指标
    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        
        print(f"\n交叉验证结果:")
        print(f"平均收益率: {avg_return:.2%}")
        print(f"平均夏普比率: {avg_sharpe:.2f}")
    
    return results