import os
import uuid
import joblib
import logging
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

logger = logging.getLogger('models')
logger.setLevel(logging.INFO)

db = SQLAlchemy()

class MLModel(db.Model):
    id = db.Column(db.String, primary_key=True)
    model_type = db.Column(db.String(120))
    params = db.Column(db.JSON)
    file_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime)
    metrics = db.Column(db.JSON)

    def to_dict(self):
        """Конвертирует модель в словарь для API ответов"""
        logger.debug(f"Converting model {self.id} to dictionary")
        return {
            'id': self.id,
            'model_type': self.model_type,
            'params': self.params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metrics': self.metrics
        }

# Доступные модели
AVAILABLE_MODELS = {
    'random_forest': {
        'class': RandomForestClassifier,
        'hyperparameters': ['n_estimators', 'max_depth', 'random_state'],
        'description': 'Random Forest Classifier'
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'hyperparameters': ['C', 'solver', 'max_iter'],
        'description': 'Logistic Regression'
    }
}

def get_model_path(model_id):
    """Возвращает путь к файлу модели"""
    logger.debug(f"Getting model path for model ID: {model_id}")
    os.makedirs("saved_models", exist_ok=True)
    path = f"saved_models/{model_id}.joblib"
    logger.debug(f"Model path: {path}")
    return path

def convert_params(params):
    """Конвертирует строковые параметры в правильные типы"""
    logger.debug(f"Converting parameters: {params}")
    converted_params = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value.isdigit():
                converted_params[key] = int(value)
                logger.debug(f"Converted parameter {key} to int: {value}")
            else:
                try:
                    converted_params[key] = float(value)
                    logger.debug(f"Converted parameter {key} to float: {value}")
                except ValueError:
                    converted_params[key] = value
                    logger.debug(f"Parameter {key} kept as string: {value}")
        else:
            converted_params[key] = value
            logger.debug(f"Parameter {key} kept as original type: {type(value)}")
    
    logger.info(f"Parameters conversion completed. Converted {len(converted_params)} parameters")
    return converted_params

def calculate_metrics(y_true, y_pred):
    """Вычисляет метрики модели"""
    logger.debug(f"Calculating metrics for {len(y_true)} samples")
    try:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        
        logger.info(f"Metrics calculated - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Возвращаем метрики по умолчанию в случае ошибки
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }

def create_model_record(model_id, model_type, params, file_path, metrics):
    """Создает запись модели в БД"""
    logger.info(f"Creating model record: ID={model_id}, Type={model_type}")
    logger.debug(f"Model params: {params}, Metrics: {metrics}")
    
    record = MLModel(
        id=model_id,
        model_type=model_type,
        params=params,
        file_path=file_path,
        created_at=datetime.now(),
        metrics=metrics
    )
    
    logger.debug(f"Model record created successfully: {model_id}")
    return record