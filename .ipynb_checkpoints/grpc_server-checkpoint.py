import os
from flask import Flask, redirect, url_for, session, request, jsonify
from flask_restx import Api, Resource, Namespace, fields, abort
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

# Импорт из models.py
from models import db, MLModel, AVAILABLE_MODELS, get_model_path, convert_params, calculate_metrics, create_model_record
import uuid
import joblib

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Инициализация БД
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

# OAuth конфигурация (остается без изменений)
oauth = OAuth(app)
github = oauth.register(
    name='github',
    client_id='Ov23liWccuX6xt5sYPwl',
    client_secret='9594d99c778b94ad8b9441937d182badb43ae795',
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

@app.route("/") 
def index(): 
    return redirect("/login")

@app.route('/login')
def registro():
    github = oauth.create_client('github')
    redirect_uri = url_for('authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    resp = github.get('user', token=token)
    profile = resp.json()
    if 'id' not in profile:
        abort(400, "GitHub authorization failed")
    session['token_oauth'] = token
    session['github_id'] = profile['id']
    return redirect(url_for('index'))

def get_user_id():
    return session.get('github_id')

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

# Endpoints (упрощенные за счет вынесенной логики)
@namespace.route('/health')
class Health(Resource):
    @api.doc(description="Проверка статуса сервиса")
    def get(self):
        return {'status': 'ok'}, 200

@namespace.route('/model-classes')
class ModelClasses(Resource):
    @api.doc(description="List of models available and their parameters")
    def get(self):
        models_info = {}
        for key, val in AVAILABLE_MODELS.items():
            models_info[key] = {
                "class_name": val["class"].__name__,
                "hyperparameters": val["hyperparameters"],
                "description": val["description"]
            }
        return models_info, 200

@namespace.route('/models/train')
class TrainModel(Resource):
    @api.doc(description="Model training")
    @api.expect(train_model)
    def post(self):
        data = request.get_json()
        model_type = data.get('model_type')
        params = data.get('params', {})
        X = data.get('X')
        y = data.get('y')

        if model_type not in AVAILABLE_MODELS:
            abort(400, 'Unsupported model type')

        # Используем функции из models.py
        converted_params = convert_params(params)
        ModelClass = AVAILABLE_MODELS[model_type]['class']
        model = ModelClass(**converted_params)
        model.fit(X, y)

        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)

        model_id = str(uuid.uuid4())
        path = get_model_path(model_id)
        joblib.dump(model, path)

        record = create_model_record(model_id, model_type, converted_params, path, metrics)
        db.session.add(record)
        db.session.commit()

        return {'model_id': model_id, 'metrics': metrics}, 201

@namespace.route('/models')
class ListModels(Resource):
    @api.doc(description="Get list of models trained")
    def get(self):
        models = MLModel.query.all()
        return [model.to_dict() for model in models], 200

@namespace.route('/models/<string:model_id>')
class ModelById(Resource):
    @api.doc(description="Get information on a trained model")
    def get(self, model_id):
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            abort(404, 'Model not found')
        return record.to_dict(), 200

    @api.doc(description="Delete model")
    def delete(self, model_id):
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            abort(404, 'Model not found')
        if os.path.exists(record.file_path):
            os.remove(record.file_path)
        db.session.delete(record)
        db.session.commit()
        return '', 204

@namespace.route('/models/<string:model_id>/predict')
class ModelPredict(Resource):
    @api.doc(description="Make prediction")
    @api.expect(predict_model)
    def post(self, model_id):
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            abort(404, 'Model not found')
        model = joblib.load(record.file_path)
        X = request.json.get('X')
        preds = model.predict(X).tolist()
        return {'predictions': preds}, 200

@namespace.route('/models/<string:model_id>/retrain')
class ModelRetrain(Resource):
    @api.doc(description="Retrain existing model")
    @api.expect(retrain_model)
    def post(self, model_id):
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            abort(404, 'Model not found')

        model = joblib.load(record.file_path)
        X = request.json.get('X')
        y = request.json.get('y')

        model.fit(X, y)
        joblib.dump(model, record.file_path)

        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)
        record.metrics = metrics
        db.session.commit()

        return {'status': 'retrained', 'metrics': metrics}, 200

@namespace.route('/metrics/<string:model_id>')
class ModelMetrics(Resource):
    @api.doc(description="Get model scores")
    def get(self, model_id):
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            abort(404, 'Model not found')
        return record.metrics, 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)