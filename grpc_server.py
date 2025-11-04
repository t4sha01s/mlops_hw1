import grpc
from concurrent import futures
import app_pb2
import app_pb2_grpc
import joblib
import uuid
import os
import logging
from logging.handlers import RotatingFileHandler
from models import db, MLModel, AVAILABLE_MODELS, get_model_path, convert_params, calculate_metrics, create_model_record
from flask import Flask

# Настройка логгера для gRPC сервера
logger = logging.getLogger('grpc_server')
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    'logs/grpc_server.log', 
    maxBytes=1024 * 1024,
    backupCount=10
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db.init_app(app)

class MLService(app_pb2_grpc.MLServiceServicer):
    
    def HealthCheck(self, request, context):
        logger.info("Health check requested via gRPC")
        return app_pb2.HealthResponse(status="ok")
    
    def GetModelClasses(self, request, context):
        logger.info("Request for available model classes via gRPC")
        model_classes = {}
        for key, val in AVAILABLE_MODELS.items():
            model_classes[key] = app_pb2.ModelClassInfo(
                class_name=val["class"].__name__,
                hyperparameters=val["hyperparameters"],
                description=val["description"]
            )
        logger.info(f"Returning {len(model_classes)} model classes via gRPC")
        return app_pb2.ModelClassesResponse(model_classes=model_classes)
    
    def ListModels(self, request, context):
        logger.info("Request for list of all models via gRPC")
        with app.app_context():
            models = MLModel.query.all()
            model_list = []
            for m in models:
                model_response = app_pb2.ModelResponse(
                    id=str(m.id),
                    model_type=str(m.model_type),
                    params={str(k): str(v) for k, v in m.params.items()} if m.params else {},
                    created_at=m.created_at.isoformat() if m.created_at else "",
                    metrics={str(k): float(v) for k, v in m.metrics.items()} if m.metrics else {}
                )
                model_list.append(model_response)
            logger.info(f"Returning {len(model_list)} models via gRPC")
            return app_pb2.ListModelsResponse(models=model_list)
    
    def TrainModel(self, request, context):
        logger.info("Starting model training request via gRPC")
        with app.app_context():
            model_type = request.model_type
            params = dict(request.params)
            X = [list(row.features) for row in request.X]
            y = list(request.y)

            logger.info(f"Training model type: {model_type} with {len(X)} samples via gRPC")

            if model_type not in AVAILABLE_MODELS:
                logger.error(f"Unsupported model type via gRPC: {model_type}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported model type")

            converted_params = convert_params(params)
            logger.debug(f"Converted parameters via gRPC: {converted_params}")
            
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

            logger.info(f"Model trained successfully via gRPC. ID: {model_id}, Metrics: {metrics}")
            return app_pb2.TrainResponse(
                model_id=model_id, 
                metrics={k: float(v) for k, v in metrics.items()}
            )

    def GetModel(self, request, context):
        logger.info(f"Request for model info via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            logger.info(f"Returning model info via gRPC: {request.model_id}")
            return app_pb2.ModelResponse(
                id=str(record.id),
                model_type=str(record.model_type),
                params={str(k): str(v) for k, v in record.params.items()} if record.params else {},
                created_at=record.created_at.isoformat() if record.created_at else "",
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )
    
    def DeleteModel(self, request, context):
        logger.info(f"Request to delete model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for deletion via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            if os.path.exists(record.file_path):
                os.remove(record.file_path)
                logger.info(f"Model file deleted via gRPC: {record.file_path}")
            
            db.session.delete(record)
            db.session.commit()
            
            logger.info(f"Model deleted successfully via gRPC: {request.model_id}")
            return app_pb2.DeleteResponse(success=True)

    def Predict(self, request, context):
        logger.info(f"Prediction request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for prediction via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

            model = joblib.load(record.file_path)
            X = [list(row.features) for row in request.X]
            logger.info(f"Making prediction via gRPC with {len(X)} samples")
            
            preds = model.predict(X).tolist()
            logger.info(f"Prediction completed via gRPC. Returning {len(preds)} predictions")
            return app_pb2.PredictResponse(predictions=[float(p) for p in preds])
    
    def RetrainModel(self, request, context):
        logger.info(f"Retrain request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for retraining via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

            model = joblib.load(record.file_path)
            X = [list(row.features) for row in request.X]
            y = list(request.y)
            logger.info(f"Retraining model via gRPC with {len(X)} samples")

            model.fit(X, y)
            joblib.dump(model, record.file_path)

            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)
            record.metrics = metrics
            db.session.commit()

            logger.info(f"Model retrained successfully via gRPC: {request.model_id}, New metrics: {metrics}")
            return app_pb2.RetrainResponse(
                metrics={k: float(v) for k, v in metrics.items()}
            )

    def GetMetrics(self, request, context):
        logger.info(f"Metrics request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for metrics via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            logger.info(f"Returning metrics via gRPC for model: {request.model_id}")
            return app_pb2.MetricsResponse(
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )

def serve():
    logger.info("Starting gRPC server")
    with app.app_context():
        db.create_all()
        logger.info("Database tables created for gRPC server")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    app_pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("gRPC server started on port 50051")
    print("gRPC server started on port 50051")
    server.start()
    logger.info("gRPC server waiting for termination")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()