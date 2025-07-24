"""
Reinforcement Learning model evaluation and backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .environment import TradingEnvironment
from ..config.settings import settings


class RLEvaluator:
    """
    Comprehensive evaluation system for RL trading models.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        logger.info("RLEvaluator initialized")
    
    def evaluate_model_performance(
        self,
        model,
        test_data: pd.DataFrame,
        model_name: str = "RL_Model",
        n_episodes: int = 1
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RL model performance.
        
        Args:
            model: Trained RL model
            test_data: Test dataset
            model_name: Name of the model
            n_episodes: Number of episodes to run
        
        Returns:
            Comprehensive evaluation results
        """
        try:
            logger.info(f"Evaluating {model_name} performance on {len(test_data)} samples")
            
            # Create test environment
            env = TradingEnvironment(
                test_data,
                initial_balance=settings.trading.initial_capital,
                transaction_cost=settings.trading.transaction_cost,
                max_position=settings.trading.max_position_size
            )
            
            results = []
            
            for episode in range(n_episodes):
                episode_result = self._run_single_episode(model, env, episode)
                results.append(episode_result)
            
            # Aggregate results
            aggregated_results = self._aggregate_episode_results(results, model_name)
            
            # Calculate additional metrics
            aggregated_results.update(self._calculate_risk_metrics(results))
            
            # Store results
            self.evaluation_results[model_name] = aggregated_results
            
            logger.info(f"Evaluation completed for {model_name}")
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def _run_single_episode(
        self, 
        model, 
        env: TradingEnvironment, 
        episode_num: int
    ) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        
        obs, _ = env.reset()
        done = False
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'positions': [],
            'prices': [],
            'timestamps': []
        }
        
        step = 0
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, _, info = env.step(action)
            
            # Record data
            episode_data['actions'].append(int(action))
            episode_data['rewards'].append(reward)
            episode_data['portfolio_values'].append(info['portfolio_value'])
            episode_data['positions'].append(info['position'])
            episode_data['prices'].append(info['current_price'])
            episode_data['timestamps'].append(step)
            
            step += 1
        
        # Calculate episode metrics
        portfolio_values = np.array(episode_data['portfolio_values'])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        episode_metrics = {
            'episode': episode_num,
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'final_portfolio_value': portfolio_values[-1],
            'max_portfolio_value': np.max(portfolio_values),
            'min_portfolio_value': np.min(portfolio_values),
            'volatility': np.std(returns) if len(returns) > 1 else 0,
            'sharpe_ratio': info.get('sharpe_ratio', 0),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'num_trades': info.get('num_trades', 0),
            'num_steps': step,
            'episode_data': episode_data
        }
        
        return episode_metrics
    
    def _aggregate_episode_results(
        self, 
        results: List[Dict[str, Any]], 
        model_name: str
    ) -> Dict[str, Any]:
        """Aggregate results from multiple episodes."""
        
        if not results:
            return {}
        
        # Extract metrics
        total_returns = [r['total_return'] for r in results]
        final_values = [r['final_portfolio_value'] for r in results]
        volatilities = [r['volatility'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        num_trades = [r['num_trades'] for r in results]
        
        aggregated = {
            'model_name': model_name,
            'n_episodes': len(results),
            
            # Return metrics
            'mean_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'min_total_return': np.min(total_returns),
            'max_total_return': np.max(total_returns),
            
            # Portfolio value metrics
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            
            # Risk metrics
            'mean_volatility': np.mean(volatilities),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            
            # Trading metrics
            'mean_num_trades': np.mean(num_trades),
            'win_rate': sum(1 for r in total_returns if r > 0) / len(total_returns),
            
            # Consistency metrics
            'return_consistency': 1 - (np.std(total_returns) / (abs(np.mean(total_returns)) + 1e-8)),
            'positive_episodes': sum(1 for r in total_returns if r > 0),
            
            # Raw results for detailed analysis
            'episode_results': results
        }
        
        return aggregated
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    def _calculate_risk_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate additional risk metrics."""
        
        try:
            all_returns = []
            all_portfolio_values = []
            
            for result in results:
                episode_data = result['episode_data']
                portfolio_values = np.array(episode_data['portfolio_values'])
                
                if len(portfolio_values) > 1:
                    returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    all_returns.extend(returns)
                    all_portfolio_values.extend(portfolio_values)
            
            if not all_returns:
                return {}
            
            all_returns = np.array(all_returns)
            all_portfolio_values = np.array(all_portfolio_values)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(all_returns, 5)
            var_99 = np.percentile(all_returns, 1)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(all_returns[all_returns <= var_95])
            cvar_99 = np.mean(all_returns[all_returns <= var_99])
            
            # Calmar Ratio
            annual_return = np.mean(all_returns) * 252 * 24  # Assuming hourly data
            max_dd = self._calculate_max_drawdown(all_portfolio_values)
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
            
            # Sortino Ratio
            downside_returns = all_returns[all_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = annual_return / downside_std if downside_std != 0 else 0
            
            # Omega Ratio (simplified)
            threshold = 0.0
            gains = all_returns[all_returns > threshold]
            losses = all_returns[all_returns <= threshold]
            omega_ratio = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 else float('inf')
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'omega_ratio': min(omega_ratio, 10.0),  # Cap at 10 for display
                'skewness': float(pd.Series(all_returns).skew()),
                'kurtosis': float(pd.Series(all_returns).kurtosis())
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def compare_models(
        self, 
        models: Dict[str, Any], 
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare multiple RL models.
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test dataset
        
        Returns:
            Comparison results
        """
        try:
            logger.info(f"Comparing {len(models)} models")
            
            comparison_results = {}
            
            # Evaluate each model
            for model_name, model in models.items():
                results = self.evaluate_model_performance(model, test_data, model_name)
                comparison_results[model_name] = results
            
            # Create comparison summary
            summary = self._create_comparison_summary(comparison_results)
            
            return {
                'individual_results': comparison_results,
                'summary': summary,
                'best_model': summary.get('best_overall', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def _create_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary comparison of models."""
        
        if not results:
            return {}
        
        metrics = ['mean_total_return', 'mean_sharpe_ratio', 'mean_max_drawdown', 'win_rate', 'mean_volatility']
        
        summary = {}
        
        for metric in metrics:
            metric_values = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    metric_values[model_name] = model_results[metric]
            
            if metric_values:
                if metric == 'mean_max_drawdown':  # Lower is better
                    best_model = min(metric_values, key=metric_values.get)
                    worst_model = max(metric_values, key=metric_values.get)
                else:  # Higher is better
                    best_model = max(metric_values, key=metric_values.get)
                    worst_model = min(metric_values, key=metric_values.get)
                
                summary[f'best_{metric}'] = best_model
                summary[f'worst_{metric}'] = worst_model
                summary[f'{metric}_values'] = metric_values
        
        # Overall best model (weighted score)
        if results:
            model_scores = {}
            
            for model_name in results.keys():
                score = 0
                
                # Return (40% weight)
                if 'mean_total_return' in results[model_name]:
                    score += results[model_name]['mean_total_return'] * 0.4
                
                # Sharpe ratio (30% weight)
                if 'mean_sharpe_ratio' in results[model_name]:
                    score += results[model_name]['mean_sharpe_ratio'] * 0.3
                
                # Max drawdown penalty (20% weight)
                if 'mean_max_drawdown' in results[model_name]:
                    score -= abs(results[model_name]['mean_max_drawdown']) * 0.2
                
                # Win rate (10% weight)
                if 'win_rate' in results[model_name]:
                    score += results[model_name]['win_rate'] * 0.1
                
                model_scores[model_name] = score
            
            summary['best_overall'] = max(model_scores, key=model_scores.get)
            summary['model_scores'] = model_scores
        
        return summary
    
    def generate_performance_report(
        self, 
        model_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed performance report.
        
        Args:
            model_name: Name of the model to report on
            save_path: Optional path to save the report
        
        Returns:
            Report as string
        """
        try:
            if model_name not in self.evaluation_results:
                return f"No evaluation results found for {model_name}"
            
            results = self.evaluation_results[model_name]
            
            report = f"""
PERFORMANCE REPORT: {model_name}
{'='*50}

RETURN METRICS:
- Mean Total Return: {results.get('mean_total_return', 0):.2%}
- Standard Deviation: {results.get('std_total_return', 0):.2%}
- Best Episode Return: {results.get('max_total_return', 0):.2%}
- Worst Episode Return: {results.get('min_total_return', 0):.2%}

RISK METRICS:
- Mean Sharpe Ratio: {results.get('mean_sharpe_ratio', 0):.3f}
- Mean Max Drawdown: {results.get('mean_max_drawdown', 0):.2%}
- Worst Max Drawdown: {results.get('worst_max_drawdown', 0):.2%}
- Mean Volatility: {results.get('mean_volatility', 0):.2%}

ADVANCED RISK METRICS:
- VaR (95%): {results.get('var_95', 0):.2%}
- VaR (99%): {results.get('var_99', 0):.2%}
- CVaR (95%): {results.get('cvar_95', 0):.2%}
- Calmar Ratio: {results.get('calmar_ratio', 0):.3f}
- Sortino Ratio: {results.get('sortino_ratio', 0):.3f}

TRADING METRICS:
- Win Rate: {results.get('win_rate', 0):.2%}
- Mean Number of Trades: {results.get('mean_num_trades', 0):.1f}
- Return Consistency: {results.get('return_consistency', 0):.3f}
- Positive Episodes: {results.get('positive_episodes', 0)}/{results.get('n_episodes', 0)}

PORTFOLIO METRICS:
- Mean Final Value: ${results.get('mean_final_value', 0):,.2f}
- Standard Deviation: ${results.get('std_final_value', 0):,.2f}

STATISTICAL PROPERTIES:
- Skewness: {results.get('skewness', 0):.3f}
- Kurtosis: {results.get('kurtosis', 0):.3f}
- Omega Ratio: {results.get('omega_ratio', 0):.3f}

{'='*50}
Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
                logger.info(f"Performance report saved to {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {str(e)}"
    
    def plot_performance_comparison(
        self, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot performance comparison of evaluated models.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('RL Model Performance Comparison', fontsize=16)
            
            # Extract data for plotting
            model_names = list(self.evaluation_results.keys())
            
            # Plot 1: Total Returns
            returns = [self.evaluation_results[name].get('mean_total_return', 0) for name in model_names]
            return_stds = [self.evaluation_results[name].get('std_total_return', 0) for name in model_names]
            
            axes[0, 0].bar(model_names, returns, yerr=return_stds, capsize=5)
            axes[0, 0].set_title('Mean Total Return')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Sharpe Ratios
            sharpe_ratios = [self.evaluation_results[name].get('mean_sharpe_ratio', 0) for name in model_names]
            
            axes[0, 1].bar(model_names, sharpe_ratios)
            axes[0, 1].set_title('Mean Sharpe Ratio')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Max Drawdowns
            max_drawdowns = [abs(self.evaluation_results[name].get('mean_max_drawdown', 0)) for name in model_names]
            
            axes[1, 0].bar(model_names, max_drawdowns, color='red', alpha=0.7)
            axes[1, 0].set_title('Mean Max Drawdown')
            axes[1, 0].set_ylabel('Max Drawdown')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Win Rates
            win_rates = [self.evaluation_results[name].get('win_rate', 0) for name in model_names]
            
            axes[1, 1].bar(model_names, win_rates, color='green', alpha=0.7)
            axes[1, 1].set_title('Win Rate')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance comparison plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting performance comparison: {e}")
    
    def export_results_to_csv(self, filepath: str) -> bool:
        """
        Export evaluation results to CSV.
        
        Args:
            filepath: Path to save CSV file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results to export")
                return False
            
            # Prepare data for CSV
            rows = []
            
            for model_name, results in self.evaluation_results.items():
                row = {'model_name': model_name}
                
                # Add all numeric metrics
                for key, value in results.items():
                    if isinstance(value, (int, float)) and key != 'n_episodes':
                        row[key] = value
                
                rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Evaluation results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results to CSV: {e}")
            return False