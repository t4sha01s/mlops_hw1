import grpc
from concurrent import futures
import app_pb2
import app_pb2_grpc
import joblib
import uuid
import os

# Импорт из models.py
from models import db, MLModel, AVAILABLE_MODELS, get_model_path, convert_params, calculate_metrics, create_model_record
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db.init_app(app)

class MLService(app_pb2_grpc.MLServiceServicer):
    
    def HealthCheck(self, request, context):
        return app_pb2.HealthResponse(status="ok")
    
    def GetModelClasses(self, request, context):
        model_classes = {}
        for key, val in AVAILABLE_MODELS.items():
            model_classes[key] = app_pb2.ModelClassInfo(
                class_name=val["class"].__name__,
                hyperparameters=val["hyperparameters"],
                description=val["description"]
            )
        return app_pb2.ModelClassesResponse(model_classes=model_classes)
    
    def ListModels(self, request, context):
        with app.app_context():
            models = MLModel.query.all()
            model_list = []
            for m in models:
                # Убедимся, что все данные правильных типов
                model_response = app_pb2.ModelResponse(
                    id=str(m.id),
                    model_type=str(m.model_type),
                    params={str(k): str(v) for k, v in m.params.items()} if m.params else {},
                    created_at=m.created_at.isoformat() if m.created_at else "",
                    metrics={str(k): float(v) for k, v in m.metrics.items()} if m.metrics else {}
                )
                model_list.append(model_response)
            return app_pb2.ListModelsResponse(models=model_list)
    
    def TrainModel(self, request, context):
        with app.app_context():
            model_type = request.model_type
            params = dict(request.params)
            X = [list(row.features) for row in request.X]
            y = list(request.y)

            if model_type not in AVAILABLE_MODELS:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported model type")

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

            return app_pb2.TrainResponse(
                model_id=model_id, 
                metrics={k: float(v) for k, v in metrics.items()}
            )

    def GetModel(self, request, context):
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            return app_pb2.ModelResponse(
                id=str(record.id),
                model_type=str(record.model_type),
                params={str(k): str(v) for k, v in record.params.items()} if record.params else {},
                created_at=record.created_at.isoformat() if record.created_at else "",
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )
    
    def DeleteModel(self, request, context):
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            if os.path.exists(record.file_path):
                os.remove(record.file_path)
            
            db.session.delete(record)
            db.session.commit()
            
            return app_pb2.DeleteResponse(success=True)

    def Predict(self, request, context):
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

            model = joblib.load(record.file_path)
            X = [list(row.features) for row in request.X]
            preds = model.predict(X).tolist()
            return app_pb2.PredictResponse(predictions=[float(p) for p in preds])
    
    def RetrainModel(self, request, context):
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

            model = joblib.load(record.file_path)
            X = [list(row.features) for row in request.X]
            y = list(request.y)

            model.fit(X, y)
            joblib.dump(model, record.file_path)

            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)
            record.metrics = metrics
            db.session.commit()

            return app_pb2.RetrainResponse(
                metrics={k: float(v) for k, v in metrics.items()}
            )

    def GetMetrics(self, request, context):
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            return app_pb2.MetricsResponse(
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )

def serve():
    with app.app_context():
        db.create_all()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    app_pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()