"""
Machine Learning forecasting models including LSTM and Transformers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import settings


class MLForecaster:
    """
    Advanced machine learning forecasting system with multiple models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.sequence_length = 60
        
        logger.info("MLForecaster initialized")
    
    def prepare_data_for_ml(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for machine learning models.
        
        Args:
            data: DataFrame with features
            target_column: Column to predict
            feature_columns: List of feature columns to use
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            logger.info("Preparing data for ML models")
            
            df = data.copy()
            
            # Auto-select feature columns if not provided
            if feature_columns is None:
                # Exclude non-numeric and target columns
                exclude_cols = [target_column, 'Symbol', 'Market_Regime', 'Vol_Regime', 'Trend_Regime', 'Mom_Regime']
                feature_columns = [col for col in df.columns 
                                 if df[col].dtype in ['float64', 'int64'] 
                                 and col not in exclude_cols]
            
            # Ensure feature columns exist
            feature_columns = [col for col in feature_columns if col in df.columns]
            
            if not feature_columns:
                raise ValueError("No valid feature columns found")
            
            # Remove rows with NaN values
            df = df[feature_columns + [target_column]].dropna()
            
            X = df[feature_columns].values
            y = df[target_column].values
            
            logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing data for ML: {e}")
            return np.array([]), np.array([]), []
    
    def create_lstm_model(
        self, 
        input_shape: Tuple[int, int],
        output_dim: int = 1
    ) -> tf.keras.Model:
        """
        Create LSTM model for time series forecasting.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            output_dim: Number of output dimensions
        
        Returns:
            Compiled LSTM model
        """
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(output_dim)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"LSTM model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None
    
    def create_transformer_model(
        self, 
        input_shape: Tuple[int, int],
        output_dim: int = 1,
        num_heads: int = 8,
        ff_dim: int = 128
    ) -> tf.keras.Model:
        """
        Create Transformer model for time series forecasting.
        
        Args:
            input_shape: Shape of input sequences
            output_dim: Number of output dimensions
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
        
        Returns:
            Compiled Transformer model
        """
        try:
            inputs = Input(shape=input_shape)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=input_shape[-1]
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
            
            # Feed Forward
            ffn_output = Dense(ff_dim, activation='relu')(attention_output)
            ffn_output = Dense(input_shape[-1])(ffn_output)
            
            # Add & Norm
            ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
            
            # Output layers
            outputs = Dense(64, activation='relu')(pooled)
            outputs = Dropout(0.2)(outputs)
            outputs = Dense(output_dim)(outputs)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Transformer model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating Transformer model: {e}")
            return None
    
    def train_lstm_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train LSTM model on time series data.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training LSTM model")
            
            # Create model
            model = self.create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            if model is None:
                return {'success': False, 'error': 'Failed to create model'}
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model
            self.models['lstm'] = model
            
            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results = {
                'success': True,
                'model_type': 'LSTM',
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None
            }
            
            logger.info(f"LSTM training completed. MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_transformer_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train Transformer model on time series data.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Transformer model")
            
            # Create model
            model = self.create_transformer_model(input_shape=(X.shape[1], X.shape[2]))
            if model is None:
                return {'success': False, 'error': 'Failed to create model'}
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model
            self.models['transformer'] = model
            
            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results = {
                'success': True,
                'model_type': 'Transformer',
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None
            }
            
            logger.info(f"Transformer training completed. MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_ensemble_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train ensemble of traditional ML models.
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training ensemble models")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            models_to_train = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            results = {}
            
            for name, model in models_to_train.items():
                try:
                    # Cross-validation scores
                    cv_scores = []
                    
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)
                        cv_scores.append(score)
                    
                    # Train on full dataset
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    # Store model and results
                    self.models[name] = model
                    
                    results[name] = {
                        'cv_score_mean': float(np.mean(cv_scores)),
                        'cv_score_std': float(np.std(cv_scores)),
                        'mse': float(mean_squared_error(y, y_pred)),
                        'mae': float(mean_absolute_error(y, y_pred)),
                        'r2': float(r2_score(y, y_pred))
                    }
                    
                    logger.info(f"{name} trained. R²: {results[name]['r2']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            return {'success': True, 'models': results}
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_prophet_model(self, data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """
        Train Prophet model for time series forecasting.
        
        Args:
            data: DataFrame with datetime index and target column
            target_column: Column to forecast
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Prophet model")
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data[target_column]
            })
            
            # Create and train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            # Make predictions on training data
            forecast = model.predict(prophet_data)
            
            # Calculate metrics
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Store model
            self.models['prophet'] = model
            
            results = {
                'success': True,
                'model_type': 'Prophet',
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            logger.info(f"Prophet training completed. MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(
        self, 
        model_name: str, 
        X: np.ndarray,
        steps_ahead: int = 1
    ) -> np.ndarray:
        """
        Make predictions using specified model.
        
        Args:
            model_name: Name of the model to use
            X: Input data
            steps_ahead: Number of steps to predict ahead
        
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            if model_name in ['lstm', 'transformer']:
                predictions = model.predict(X)
            elif model_name == 'prophet':
                # For Prophet, X should be a DataFrame with 'ds' column
                future = model.make_future_dataframe(periods=steps_ahead, freq='H')
                forecast = model.predict(future)
                predictions = forecast['yhat'].values[-steps_ahead:]
            else:
                predictions = model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {e}")
            return np.array([])
    
    def ensemble_predict(self, X: np.ndarray, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            X: Input data
            weights: Optional weights for each model
        
        Returns:
            Ensemble predictions
        """
        try:
            if not self.models:
                raise ValueError("No trained models available")
            
            predictions = {}
            
            # Get predictions from all available models
            for name, model in self.models.items():
                try:
                    if name in ['lstm', 'transformer']:
                        pred = model.predict(X)
                    elif name == 'prophet':
                        continue  # Skip Prophet for ensemble (needs different input format)
                    else:
                        pred = model.predict(X.reshape(X.shape[0], -1))  # Flatten for sklearn models
                    
                    predictions[name] = pred.flatten()
                    
                except Exception as e:
                    logger.warning(f"Error getting predictions from {name}: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No valid predictions obtained")
            
            # Calculate ensemble prediction
            if weights is None:
                # Equal weights
                weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
            
            ensemble_pred = np.zeros(len(list(predictions.values())[0]))
            
            for name, pred in predictions.items():
                weight = weights.get(name, 0)
                ensemble_pred += weight * pred
            
            logger.info(f"Ensemble prediction completed using {len(predictions)} models")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])
    
    def save_models(self, filepath: str) -> bool:
        """
        Save all trained models to disk.
        
        Args:
            filepath: Base filepath for saving models
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for name, model in self.models.items():
                if name in ['lstm', 'transformer']:
                    model.save(f"{filepath}_{name}.h5")
                elif name == 'prophet':
                    joblib.dump(model, f"{filepath}_{name}.pkl")
                else:
                    joblib.dump(model, f"{filepath}_{name}.pkl")
            
            # Save feature columns
            joblib.dump(self.feature_columns, f"{filepath}_features.pkl")
            
            logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            filepath: Base filepath for loading models
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            
            # Load feature columns
            if os.path.exists(f"{filepath}_features.pkl"):
                self.feature_columns = joblib.load(f"{filepath}_features.pkl")
            
            # Load models
            model_files = {
                'lstm': f"{filepath}_lstm.h5",
                'transformer': f"{filepath}_transformer.h5",
                'prophet': f"{filepath}_prophet.pkl",
                'random_forest': f"{filepath}_random_forest.pkl",
                'gradient_boosting': f"{filepath}_gradient_boosting.pkl",
                'linear_regression': f"{filepath}_linear_regression.pkl"
            }
            
            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        if name in ['lstm', 'transformer']:
                            self.models[name] = tf.keras.models.load_model(file_path)
                        else:
                            self.models[name] = joblib.load(file_path)
                        logger.info(f"Loaded {name} model")
                    except Exception as e:
                        logger.warning(f"Error loading {name} model: {e}")
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False