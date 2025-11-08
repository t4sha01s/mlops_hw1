import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, redirect, url_for, session, request
from flask_restx import Api, Resource, Namespace, fields, abort
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
from models import db, MLModel, AVAILABLE_MODELS, get_model_path, convert_params, calculate_metrics, create_model_record
import joblib
import uuid
from datetime import datetime

"""
Setting app configurations
"""

# Настройка логгера для Flask приложения
logger = logging.getLogger('flask_app')
logger.setLevel(logging.INFO)

# Файловый обработчик
file_handler = RotatingFileHandler(
    'logs/flask_api.log', 
    maxBytes=1024 * 1024,  
    backupCount=10
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Инициализация базы данных
db.init_app(app)

api = Api(
    app, 
    title='REST API models depl',
    description="""Documentation for models delp.\n\n
    Before getting to work with this application please go through authorization procedure.
    To do so add /login to your current url.
    (for example, http://localhost:5000/login)"""
 )

namespace = api.namespace('', 'Click the down arrow to expand the content')

"""
Initializing authorization via github
"""

oauth = OAuth(app)
github = oauth.register(
    name='github',
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

@app.route("/") 
def index(): 
    logger.info("Redirecting from index to login page")
    return redirect("/login")

@app.route('/login')
def registro():
    logger.info("Starting OAuth login process")
    github = oauth.create_client('github')
    redirect_uri = url_for('authorize', _external=True)
    logger.info(f"Redirecting to GitHub OAuth: {redirect_uri}")
    return github.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    logger.info("Processing OAuth authorization callback")
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    resp = github.get('user', token=token)
    profile = resp.json()
    
    if 'id' not in profile:
        logger.error("GitHub authorization failed - no user ID in profile")
        abort(400, "GitHub authorization failed")
        
    github_id = profile['id']
    session['token_oauth'] = token
    session['github_id'] = profile['id']
    logger.info(f"Successful GitHub authorization for user ID: {github_id}")
    return redirect(url_for('index'))

def get_user_id():
    if 'github_id' in session:
        return session['github_id']
    return 

# Swagger models
train_model = api.model('TrainModel', {
    'model_type': fields.String(required=True, description='Model type (random_forest / logistic_regression)'),
    'params': fields.Raw(required=True, description='Model parameters'),
    'X': fields.List(fields.List(fields.Float), required=True, description='Features'),
    'y': fields.List(fields.Integer, required=True, description='Labels')
})

predict_model = api.model('PredictModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='Data for making predictions')
})

retrain_model = api.model('RetrainModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='Features'),
    'y': fields.List(fields.Integer, required=True, description='Labels')
})

# Endpoints

@namespace.route('/health')
class Health(Resource):
    @api.doc(description="Проверка статуса сервиса")
    def get(self):
        logger.info("Health check requested")
        return {'status': 'ok'}, 200


@namespace.route('/model-classes')
class ModelClasses(Resource):
    @api.doc(description="List of models available and their parameters")
    def get(self):
        logger.info("Request for available model classes")
        models_info = {}
        for key, val in AVAILABLE_MODELS.items():
            models_info[key] = {
                "class_name": val["class"].__name__,
                "hyperparameters": val["hyperparameters"],
                "description": val["description"]
            }
        logger.info(f"Returning {len(models_info)} model classes")
        return models_info, 200

@namespace.route('/models/train')
class TrainModel(Resource):
    @api.doc(description="Model training")
    @api.expect(train_model)
    def post(self):
        logger.info("Starting model training request")
        data = request.get_json()
        model_type = data.get('model_type')
        params = data.get('params', {})
        X = data.get('X')
        y = data.get('y')

        logger.info(f"Training model type: {model_type} with {len(X)} samples")

        if model_type not in AVAILABLE_MODELS:
            logger.error(f"Unsupported model type: {model_type}")
            abort(400, 'Unsupported model type')

        # Конвертируем параметры
        converted_params = convert_params(params)
        logger.debug(f"Converted parameters: {converted_params}")
        
        ModelClass = AVAILABLE_MODELS[model_type]['class']
        model = ModelClass(**converted_params)
        model.fit(X, y)

        # Вычисляем метрики
        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)

        # Сохраняем модель
        model_id = str(uuid.uuid4())
        path = get_model_path(model_id)
        joblib.dump(model, path)

        # Создаем запись в БД
        record = create_model_record(model_id, model_type, converted_params, path, metrics)
        db.session.add(record)
        db.session.commit()

        logger.info(f"Model trained successfully. ID: {model_id}, Metrics: {metrics}")
        return {'model_id': model_id, 'metrics': metrics}, 201


@namespace.route('/models')
class ListModels(Resource):
    @api.doc(description="Get list of models trained")
    def get(self):
        logger.info("Request for list of all models")
        models = MLModel.query.all()
        result = [model.to_dict() for model in models]
        logger.info(f"Returning {len(result)} models")
        return result, 200


@namespace.route('/models/<string:model_id>')
class ModelById(Resource):
    @api.doc(description="Get information on a trained model")
    def get(self, model_id):
        logger.info(f"Request for model info: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found: {model_id}")
            abort(404, 'Model not found')
        logger.info(f"Returning model info: {model_id}")
        return record.to_dict(), 200

    @api.doc(description="Delete model")
    def delete(self, model_id):
        logger.info(f"Request to delete model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for deletion: {model_id}")
            abort(404, 'Model not found')
        if os.path.exists(record.file_path):
            os.remove(record.file_path)
            logger.info(f"Model file deleted: {record.file_path}")
        db.session.delete(record)
        db.session.commit()
        logger.info(f"Model deleted successfully: {model_id}")
        return '', 204


@namespace.route('/models/<string:model_id>/predict')
class ModelPredict(Resource):
    @api.doc(description="Make prediction")
    @api.expect(predict_model)
    def post(self, model_id):
        logger.info(f"Prediction request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for prediction: {model_id}")
            abort(404, 'Model not found')
        
        model = joblib.load(record.file_path)
        X = request.json.get('X')
        logger.info(f"Making prediction with {len(X)} samples")
        
        preds = model.predict(X).tolist()
        logger.info(f"Prediction completed. Returning {len(preds)} predictions")
        return {'predictions': preds}, 200


@namespace.route('/models/<string:model_id>/retrain')
class ModelRetrain(Resource):
    @api.doc(description="Retrain existing model")
    @api.expect(retrain_model)
    def post(self, model_id):
        logger.info(f"Retrain request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for retraining: {model_id}")
            abort(404, 'Model not found')

        model = joblib.load(record.file_path)
        X = request.json.get('X')
        y = request.json.get('y')
        logger.info(f"Retraining model with {len(X)} samples")

        model.fit(X, y)
        joblib.dump(model, record.file_path)

        # Обновляем метрики
        y_pred = model.predict(X)
        record.metrics = calculate_metrics(y, y_pred)
        db.session.commit()

        logger.info(f"Model retrained successfully: {model_id}, New metrics: {record.metrics}")
        return {'status': 'retrained', 'metrics': record.metrics}, 200


@namespace.route('/metrics/<string:model_id>')
class ModelMetrics(Resource):
    @api.doc(description="Get model scores")
    def get(self, model_id):
        logger.info(f"Metrics request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for metrics: {model_id}")
            abort(404, 'Model not found')
        logger.info(f"Returning metrics for model: {model_id}")
        return record.metrics, 200   

if __name__ == '__main__':
    logger.info("Starting Flask application")
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
    logger.info("Flask app running in debug mode")
    app.run(debug=True)