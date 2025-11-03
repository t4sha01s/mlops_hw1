import os
import uuid
import joblib
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Инициализация SQLAlchemy
db = SQLAlchemy()

# Модель MLModel
class MLModel(db.Model):
    id = db.Column(db.String, primary_key=True)
    model_type = db.Column(db.String(120))
    params = db.Column(db.JSON)
    file_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime)
    metrics = db.Column(db.JSON)

    def to_dict(self):
        """Конвертирует модель в словарь для API ответов"""
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
    os.makedirs("saved_models", exist_ok=True)
    return f"saved_models/{model_id}.joblib"

def convert_params(params):
    """Конвертирует строковые параметры в правильные типы"""
    converted_params = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value.isdigit():
                converted_params[key] = int(value)
            else:
                try:
                    converted_params[key] = float(value)
                except ValueError:
                    converted_params[key] = value
        else:
            converted_params[key] = value
    return converted_params

def calculate_metrics(y_true, y_pred):
    """Вычисляет метрики модели"""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
    }

def create_model_record(model_id, model_type, params, file_path, metrics):
    """Создает запись модели в БД"""
    return MLModel(
        id=model_id,
        model_type=model_type,
        params=params,
        file_path=file_path,
        created_at=datetime.now(),
        metrics=metrics
    )