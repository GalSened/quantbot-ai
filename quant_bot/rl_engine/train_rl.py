"""
Reinforcement Learning trainer for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import gymnasium as gym
from stable_baselines3 import PPO, DDPG, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import optuna
from loguru import logger
import os
import joblib
from datetime import datetime

from .environment import TradingEnvironment
from ..config.settings import settings


class RLTrainer:
    """
    Reinforcement Learning trainer for trading strategies.
    """
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        self.best_params = {}
        
        logger.info("RLTrainer initialized")
    
    def prepare_training_data(
        self, 
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: Full dataset
            train_ratio: Ratio of data for training
            validation_ratio: Ratio of data for validation
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        try:
            logger.info(f"Preparing training data with {len(data)} samples")
            
            # Sort by datetime
            data = data.sort_index()
            
            # Calculate split indices
            n_samples = len(data)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + validation_ratio))
            
            train_data = data.iloc[:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
            
            logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def create_training_environment(
        self, 
        data: pd.DataFrame,
        env_params: Optional[Dict[str, Any]] = None
    ) -> TradingEnvironment:
        """
        Create trading environment for training.
        
        Args:
            data: Training data
            env_params: Environment parameters
        
        Returns:
            Trading environment
        """
        try:
            if env_params is None:
                env_params = {
                    'initial_balance': settings.trading.initial_capital,
                    'transaction_cost': settings.trading.transaction_cost,
                    'max_position': settings.trading.max_position_size,
                    'lookback_window': 20
                }
            
            env = TradingEnvironment(data, **env_params)
            logger.info("Training environment created successfully")
            
            return env
            
        except Exception as e:
            logger.error(f"Error creating training environment: {e}")
            raise
    
    def train_ppo_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train PPO model for trading.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            model_params: PPO model parameters
            training_params: Training parameters
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training PPO model")
            
            # Default parameters
            if model_params is None:
                model_params = {
                    'learning_rate': settings.rl.learning_rate,
                    'n_steps': settings.rl.n_steps,
                    'batch_size': settings.rl.batch_size,
                    'n_epochs': settings.rl.n_epochs,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5
                }
            
            if training_params is None:
                training_params = {
                    'total_timesteps': settings.rl.total_timesteps,
                    'eval_freq': settings.rl.eval_freq,
                    'save_freq': settings.rl.save_freq
                }
            
            # Create environments
            train_env = self.create_training_environment(train_data)
            val_env = self.create_training_environment(val_data)
            
            # Wrap environments
            train_env = Monitor(train_env)
            val_env = Monitor(val_env)
            
            # Create model
            model = PPO(
                'MlpPolicy',
                train_env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                **model_params
            )
            
            # Setup callbacks
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path='./models/best_ppo_model',
                log_path='./logs/',
                eval_freq=training_params['eval_freq'],
                deterministic=True,
                render=False
            )
            
            # Train model
            start_time = datetime.now()
            model.learn(
                total_timesteps=training_params['total_timesteps'],
                callback=eval_callback,
                tb_log_name="PPO"
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model.save("./models/ppo_trading_model")
            self.models['ppo'] = model
            
            # Evaluate final performance
            final_performance = self._evaluate_model(model, val_env, n_episodes=10)
            
            results = {
                'success': True,
                'model_type': 'PPO',
                'training_time': training_time,
                'total_timesteps': training_params['total_timesteps'],
                'final_performance': final_performance,
                'model_params': model_params,
                'model_path': "./models/ppo_trading_model"
            }
            
            self.training_history['ppo'] = results
            logger.info(f"PPO training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training PPO model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_ddpg_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train DDPG model for trading.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            model_params: DDPG model parameters
            training_params: Training parameters
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training DDPG model")
            
            # Default parameters
            if model_params is None:
                model_params = {
                    'learning_rate': settings.rl.learning_rate,
                    'buffer_size': 100000,
                    'learning_starts': 1000,
                    'batch_size': settings.rl.batch_size,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1
                }
            
            if training_params is None:
                training_params = {
                    'total_timesteps': settings.rl.total_timesteps,
                    'eval_freq': settings.rl.eval_freq
                }
            
            # Create environments
            train_env = self.create_training_environment(train_data)
            val_env = self.create_training_environment(val_data)
            
            # Wrap environments
            train_env = Monitor(train_env)
            val_env = Monitor(val_env)
            
            # Create model
            model = DDPG(
                'MlpPolicy',
                train_env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                **model_params
            )
            
            # Setup callbacks
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path='./models/best_ddpg_model',
                log_path='./logs/',
                eval_freq=training_params['eval_freq'],
                deterministic=True,
                render=False
            )
            
            # Train model
            start_time = datetime.now()
            model.learn(
                total_timesteps=training_params['total_timesteps'],
                callback=eval_callback,
                tb_log_name="DDPG"
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model.save("./models/ddpg_trading_model")
            self.models['ddpg'] = model
            
            # Evaluate final performance
            final_performance = self._evaluate_model(model, val_env, n_episodes=10)
            
            results = {
                'success': True,
                'model_type': 'DDPG',
                'training_time': training_time,
                'total_timesteps': training_params['total_timesteps'],
                'final_performance': final_performance,
                'model_params': model_params,
                'model_path': "./models/ddpg_trading_model"
            }
            
            self.training_history['ddpg'] = results
            logger.info(f"DDPG training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training DDPG model: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_hyperparameters(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        algorithm: str = 'ppo',
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            algorithm: RL algorithm to optimize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Optimizing {algorithm} hyperparameters with {n_trials} trials")
            
            def objective(trial):
                try:
                    # Suggest hyperparameters
                    if algorithm.lower() == 'ppo':
                        params = {
                            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                            'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
                            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
                        }
                        
                        # Create environment and model
                        env = self.create_training_environment(train_data)
                        env = Monitor(env)
                        
                        model = PPO('MlpPolicy', env, verbose=0, **params)
                        
                    elif algorithm.lower() == 'ddpg':
                        params = {
                            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                            'tau': trial.suggest_float('tau', 0.001, 0.02),
                            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000])
                        }
                        
                        # Create environment and model
                        env = self.create_training_environment(train_data)
                        env = Monitor(env)
                        
                        model = DDPG('MlpPolicy', env, verbose=0, **params)
                    
                    else:
                        raise ValueError(f"Unsupported algorithm: {algorithm}")
                    
                    # Train for shorter time during optimization
                    model.learn(total_timesteps=10000)
                    
                    # Evaluate on validation set
                    val_env = self.create_training_environment(val_data)
                    val_env = Monitor(val_env)
                    
                    performance = self._evaluate_model(model, val_env, n_episodes=5)
                    
                    # Return negative total return (Optuna minimizes)
                    return -performance['mean_total_return']
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float('inf')
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Get best parameters
            best_params = study.best_params
            best_value = -study.best_value
            
            self.best_params[algorithm] = best_params
            
            results = {
                'success': True,
                'algorithm': algorithm,
                'best_params': best_params,
                'best_performance': best_value,
                'n_trials': len(study.trials),
                'study': study
            }
            
            logger.info(f"Hyperparameter optimization completed. Best performance: {best_value:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_model(
        self, 
        model, 
        env: gym.Env, 
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained RL model
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
        
        Returns:
            Performance metrics
        """
        try:
            episode_returns = []
            episode_sharpe_ratios = []
            episode_max_drawdowns = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_return = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    episode_return += reward
                
                # Get final info
                total_return = info.get('total_return', 0)
                sharpe_ratio = info.get('sharpe_ratio', 0)
                
                episode_returns.append(total_return)
                episode_sharpe_ratios.append(sharpe_ratio)
                
                # Calculate max drawdown (simplified)
                portfolio_history = env.get_portfolio_history()
                if not portfolio_history.empty and 'portfolio_value' in portfolio_history.columns:
                    values = portfolio_history['portfolio_value'].dropna()
                    if len(values) > 0:
                        peak = values.expanding().max()
                        drawdown = (values - peak) / peak
                        max_drawdown = drawdown.min()
                        episode_max_drawdowns.append(max_drawdown)
            
            return {
                'mean_total_return': np.mean(episode_returns),
                'std_total_return': np.std(episode_returns),
                'mean_sharpe_ratio': np.mean(episode_sharpe_ratios),
                'mean_max_drawdown': np.mean(episode_max_drawdowns) if episode_max_drawdowns else 0,
                'win_rate': sum(1 for r in episode_returns if r > 0) / len(episode_returns)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {
                'mean_total_return': 0,
                'std_total_return': 0,
                'mean_sharpe_ratio': 0,
                'mean_max_drawdown': 0,
                'win_rate': 0
            }
    
    def save_training_results(self, filepath: str) -> bool:
        """
        Save training results and models.
        
        Args:
            filepath: Base filepath for saving
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save training history
            joblib.dump(self.training_history, f"{filepath}_training_history.pkl")
            
            # Save best parameters
            joblib.dump(self.best_params, f"{filepath}_best_params.pkl")
            
            # Models are already saved during training
            
            logger.info(f"Training results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
            return False
    
    def load_model(self, model_path: str, algorithm: str) -> bool:
        """
        Load trained model.
        
        Args:
            model_path: Path to model file
            algorithm: Algorithm type
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if algorithm.lower() == 'ppo':
                model = PPO.load(model_path)
            elif algorithm.lower() == 'ddpg':
                model = DDPG.load(model_path)
            elif algorithm.lower() == 'sac':
                model = SAC.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            self.models[algorithm.lower()] = model
            logger.info(f"Loaded {algorithm} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False